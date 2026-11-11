#ifndef RAMULATOR_DRAM_COMMANDS_PIM_MAC_H
#define RAMULATOR_DRAM_COMMANDS_PIM_MAC_H

#include "ramulator/dram/node.h"
#include <algorithm>

namespace Ramulator::Cmd {

template <class T>
struct PIM_MAC {
  static constexpr DRAMCommandMeta meta = {.is_column_command = true};
  static constexpr BankTarget bank_target = BankTarget::Single;

  static void action(DRAMNode* bank, int cmd, const AddrVec_t& addr_vec, Clk_t clk) {
    // Computes MAC in the SRB, no DRAM state change.
  }

  static int preq(DRAMNode* bank, int cmd, const AddrVec_t& addr_vec, Clk_t clk) {
    int target_row = addr_vec[T::Level::Row];
    
    // Check if the operands are in the SRB!
    if (std::find(bank->m_srb_rows.begin(), bank->m_srb_rows.end(), target_row) != bank->m_srb_rows.end()) {
      return cmd; // Both rows in SRB? We can MAC immediately!
    }
    
    // If not in SRB, we need to load it. 
    // Wait, loading to SRB requires the bank to be Closed (our auto-precharge logic)
    if (bank->m_state == T::State::Opened || bank->m_state == T::State::PIM_Loaded) {
      return T::Command::PREpb;
    }
    
    return T::Command::PIM_ACT_SRB;
  }
};

}  // namespace Ramulator::Cmd

#endif  // RAMULATOR_DRAM_COMMANDS_PIM_MAC_H
