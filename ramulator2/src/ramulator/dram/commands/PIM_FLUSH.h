#ifndef RAMULATOR_DRAM_COMMANDS_PIM_FLUSH_H
#define RAMULATOR_DRAM_COMMANDS_PIM_FLUSH_H

#include "ramulator/dram/node.h"
#include <algorithm>

namespace Ramulator::Cmd {

template <class T>
struct PIM_FLUSH {
  static constexpr DRAMCommandMeta meta = {.is_column_command = true};
  static constexpr BankTarget bank_target = BankTarget::Single;

  static void action(DRAMNode* bank, int cmd, const AddrVec_t& addr_vec, Clk_t clk) {
    // Flushes SRB back to the DRAM array.
    // For simplicity, we just model it taking time. It doesn't permanently change bank state.
  }

  static int preq(DRAMNode* bank, int cmd, const AddrVec_t& addr_vec, Clk_t clk) {
    // Actually writing to the DRAM array requires it to be opened!
    // But for our simplified SRB model, let's say it acts like an ACT+WR+PRE macro,
    // so it can just execute from a Closed state.
    if (bank->m_state == T::State::Opened || bank->m_state == T::State::PIM_Loaded) {
      return T::Command::PREpb;
    }
    
    return cmd; // Issue the FLUSH!
  }
};

}  // namespace Ramulator::Cmd

#endif  // RAMULATOR_DRAM_COMMANDS_PIM_FLUSH_H
