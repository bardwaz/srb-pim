#ifndef RAMULATOR_DRAM_COMMANDS_PIM_ACT_SRB_H
#define RAMULATOR_DRAM_COMMANDS_PIM_ACT_SRB_H

#include "ramulator/dram/node.h"
#include <algorithm>

namespace Ramulator::Cmd {

template <class T>
struct PIM_ACT_SRB {
  // We model this as an ACT + COPY + AUTO-PRE operation.
  // It is NOT is_opening because it returns the array to Closed state.
  static constexpr DRAMCommandMeta meta = {.is_opening = false, .is_row_command = true};
  static constexpr BankTarget bank_target = BankTarget::Single;

  static void action(DRAMNode* bank, int cmd, const AddrVec_t& addr_vec, Clk_t clk) {
    // Bank state remains Closed (since it auto-precharges the DRAM array).
    // We add the row to the SRB.
    int target_row = addr_vec[T::Level::Row];
    
    // If not already in SRB, add it
    if (std::find(bank->m_srb_rows.begin(), bank->m_srb_rows.end(), target_row) == bank->m_srb_rows.end()) {
      // SRB capacity: 8 rows of 8KB = 64KB SRAM per bank (matching commercial UPMEM WRAM)
      if (bank->m_srb_rows.size() >= 8) {
        bank->m_srb_rows.erase(bank->m_srb_rows.begin()); // FIFO eviction
      }
      bank->m_srb_rows.push_back(target_row);
    }
  }

  static int preq(DRAMNode* bank, int cmd, const AddrVec_t& addr_vec, Clk_t clk) {
    switch (bank->m_state) {
      case T::State::Closed:
        return T::Command::PIM_ACT_SRB; // We can directly copy to SRB if closed
      case T::State::Opened:
        return T::Command::PREpb; // If another row is open in the sense amps, close it first!
      case T::State::PIM_Loaded:
        return T::Command::PREpb; // Legacy state, just in case
      default:
        return T::Command::PREpb;
    }
  }
};

}  // namespace Ramulator::Cmd

#endif  // RAMULATOR_DRAM_COMMANDS_PIM_ACT_SRB_H
