#include <fmt/format.h>

#include "ramulator/base/param.h"
#include "ramulator/frontend/i_frontend.h"

namespace Ramulator {

// Streaming PIM GEMV Frontend: y[M] = A[M x N] * x[N]
//
// In Matrix-Vector Multiplication:
// - Matrix A is partitioned row-wise across banks: bank = (i % num_banks)
// - Vector x is broadcast and cached in each bank's SRB (Secondary Row Buffer)
// - Each bank computes dot product between its assigned rows and vector x
// - Operational intensity is strictly O(1) (~0.5 FLOP/byte), making CPU memory-bound
//
// Physical DRAM & SRB Sizing:
// - Standard DRAM row size: 8 KB (8192 bytes = 2048 float32 elements)
// - Compute tile size: 64 float32 elements = 256 bytes
// - Tiles per 8 KB row: 8192 / 256 = 32 tiles/row
// - Consecutive column tiles sharing an 8 KB row are processed in sequence,
//   achieving 31 SRB hits for every 1 row activation (100% compute saturation).
// - Maximum matrix size capped at N <= 8192 for BTP scope.
//
class PIMGemvFrontend : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, PIMGemvFrontend, "PIMGemvFrontend")

 private:
  int m_M = 1024;
  int m_N = 1024;
  int m_num_banks = 128;
  int m_num_channels = 128;
  int m_banks_per_bg = 4;
  int m_pim_mac_cycles = 32;
  int m_tiles_per_row = 32; // Default: 8 KB row (32 tiles * 256 bytes)

  // State machine indices
  int m_bank = 0;       // Current bank [0 .. num_banks-1]
  int m_local_row = 0;  // Matrix row index local to bank [0 .. rows_per_bank-1]
  int m_col_tile = 0;   // Column tile index along row [0 .. n_tiles-1]
  int m_flush_idx = 0;  // Flush counter [0 .. M-1]
  int m_phase = 0;      // 0 = MAC, 1 = FLUSH
  bool m_all_sent = false;
  bool m_finished = false;

  size_t m_total_ops = 0;
  size_t m_ops_sent = 0;
  size_t m_ops_completed = 0;
  std::function<void(Request&)> m_complete_cb;

 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_clock_ratio, unsigned int, "clock_ratio").required();
    RAMULATOR_PARSE_PARAM(m_M, int, "M").required();
    RAMULATOR_PARSE_PARAM(m_N, int, "N").required();
    RAMULATOR_PARSE_PARAM(m_num_banks, int, "num_banks").default_val(128);
    RAMULATOR_PARSE_PARAM(m_num_channels, int, "num_channels").default_val(128);
    RAMULATOR_PARSE_PARAM(m_banks_per_bg, int, "banks_per_bg").default_val(4);
    RAMULATOR_PARSE_PARAM(m_pim_mac_cycles, int, "pim_mac_cycles").default_val(32);
    RAMULATOR_PARSE_PARAM(m_tiles_per_row, int, "tiles_per_row").default_val(32);

    if (m_M > 8192 || m_N > 8192) {
      throw std::runtime_error(fmt::format(
          "PIMGemvFrontend: Matrix dimensions ({}x{}) exceed the supported BTP cap of 8192.",
          m_M, m_N));
    }

    int n_tiles = m_N / 64;
    if (n_tiles < 1) n_tiles = 1;
    m_total_ops = (size_t)m_M * n_tiles + m_M;

    size_t rows_per_bank = m_M / m_num_banks;
    if (rows_per_bank == 0) rows_per_bank = 1;
    size_t phys_rows_per_matrix_row = (n_tiles + m_tiles_per_row - 1) / m_tiles_per_row;
    size_t total_activations = rows_per_bank * phys_rows_per_matrix_row;

    m_logger.info(fmt::format(
        "PIM GEMV Frontend: M={}, N={}, Banks={}, Channels={}, MAC_cycles={}, "
        "Row_size=8KB (tiles_per_row={}), activations_per_bank={}, Total ops={}",
        m_M, m_N, m_num_banks, m_num_channels, m_pim_mac_cycles,
        m_tiles_per_row, total_activations, m_total_ops));

    m_bank = 0;
    m_local_row = 0;
    m_col_tile = 0;
    m_flush_idx = 0;
    m_phase = 0;
    m_ops_sent = 0;
    m_ops_completed = 0;
    m_complete_cb = [this](Request&) { m_ops_completed++; };
    m_all_sent = false;
    m_finished = false;
  };

  void tick() override {
    if (m_all_sent) return;

    // Dispatch concurrently across available channels in each tick
    for (int c = 0; c < m_num_channels; c++) {
      if (m_all_sent) break;

      int type_id;
      AddrVec_t addr_vec;
      generate_next(type_id, addr_vec);

      Request req(addr_vec, type_id);
      req.size_bytes = m_memory_system->get_tx_bytes();
      req.callback = m_complete_cb;
      bool sent = m_memory_system->send(req);
      if (sent) {
        m_ops_sent++;
        advance_state();
      } else {
        // Channel buffer full; wait for next cycle
        break;
      }
    }
  };

  bool is_finished() override {
    if (!m_finished && m_all_sent && m_ops_completed >= m_ops_sent) {
      m_finished = true;
      m_logger.info(fmt::format("PIM GEMV completed. Ops sent: {}, Ops completed: {}", m_ops_sent, m_ops_completed));
    }
    return m_finished;
  };

 private:
  void generate_next(int& type_id, AddrVec_t& addr_vec) {
    int ch = 0, rank = 0, bg = 0, ba = 0;

    if (m_num_channels == m_num_banks) {
      ch = m_bank;
    } else {
      int banks_per_ch = m_num_banks / m_num_channels;
      if (banks_per_ch < 1) banks_per_ch = 1;
      ch = m_bank / banks_per_ch;
      int bank_in_ch = m_bank % banks_per_ch;
      bg = bank_in_ch / m_banks_per_bg;
      ba = bank_in_ch % m_banks_per_bg;
    }

    int n_tiles = m_N / 64;
    if (n_tiles < 1) n_tiles = 1;

    if (m_phase == 0) {
      type_id = 2; // PIM (MAC)
      int matrix_row = m_bank + m_local_row * m_num_banks;
      int linear_tile = matrix_row * n_tiles + m_col_tile;
      int dram_row = linear_tile / m_tiles_per_row;
      addr_vec = {ch, rank, bg, ba, dram_row, 0};
    } else if (m_phase == 1) {
      type_id = 3; // FLUSH
      addr_vec = {ch, rank, bg, ba, m_flush_idx, 0};
    }
  }

  void advance_state() {
    int n_tiles = m_N / 64;
    if (n_tiles < 1) n_tiles = 1;

    int rows_per_bank = m_M / m_num_banks;
    if (rows_per_bank < 1) rows_per_bank = 1;

    if (m_phase == 0) {
      // Step through banks concurrently
      m_bank++;
      if (m_bank >= m_num_banks) {
        m_bank = 0;
        // All banks advance to next column tile together
        m_col_tile++;
        if (m_col_tile >= n_tiles) {
          m_col_tile = 0;
          // All banks advance to next local row together
          m_local_row++;
          if (m_local_row >= rows_per_bank) {
            m_phase = 1; // All compute finished; proceed to flush
            m_bank = 0;
          }
        }
      }
    } else if (m_phase == 1) {
      m_flush_idx++;
      m_bank = (m_bank + 1) % m_num_banks;
      if (m_flush_idx >= m_M) {
        m_all_sent = true;
      }
    }
  }
};

}  // namespace Ramulator
