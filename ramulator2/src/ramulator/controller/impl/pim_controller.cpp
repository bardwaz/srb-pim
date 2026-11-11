#include "ramulator/base/base.h"
#include "ramulator/controller/controller_base.h"
#include "ramulator/controller/refresh/i_refresh_manager.h"
#include "ramulator/controller/rowpolicy/i_row_policy.h"

namespace Ramulator {

class PIMController : public ControllerBase {
  RAMULATOR_REGISTER_IMPLEMENTATION_DERIVED(IController, PIMController, ControllerBase, "PIMController")

 public:
  ReqBuffer m_pim_buffer;

  void init() override {
    init_base();
    // Default size for PIM buffer
    m_pim_buffer.max_size = 64; 
  }

  void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
    setup_base(frontend, memory_system);
  }

  bool send(Request& req) override {
    // 2 is the type_id we'll use for PIM requests
    if (req.type_id != 2 && req.type_id != 3) {
      return ControllerBase::send(req);
    }
    
    // Process PIM request
    m_addr_mapper->apply(req);
    req.addr_vec[0] = m_channel_id;
    
    if (req.type_id >= 0 && req.type_id < m_device.m_spec->supported_requests.size()) {
      req.final_command = m_device.m_spec->supported_requests[req.type_id];
    } else {
      req.final_command = -1; 
    }
    
    req.arrive = m_clk;
    bool is_success = m_pim_buffer.enqueue(req);
    
    if (!is_success) {
      req.arrive = -1;
      return false;
    }
    
    return true;
  }

  void tick() override {
    // Common bookkeeping: clk advance, req queue stats update, completed reads draining
    tick_prologue();

    // We give refresh requests high priority in the same tick
    m_refresh->tick();

    // Pre-schedule hooks
    m_rowpolicy->pre_schedule();
    for (auto* p : m_plugins) {
      p->pre_schedule();
    }

    // Priority: active > priority > pim > read/write
    // 1. Try to schedule from active
    Candidate cand = pick_best_ready_from(m_active_buffer, {});

    // 2. If no candidate found, try to schedule from priority
    if (!cand.valid) {
      cand = pick_priority_if();
    }

    // 3. Try to schedule PIM or R/W, based on arrival time
    if (!cand.valid && m_priority_buffer.size() == 0) {
      Candidate cand_pim = pick_best_ready_from(m_pim_buffer, [&](const Request& req) {
        return !would_close_active(req);
      });
      Candidate cand_rw = pick_rw_if();

      if (cand_pim.valid && cand_rw.valid) {
        if (cand_pim.it->arrive <= cand_rw.it->arrive) {
          cand = cand_pim;
        } else {
          cand = cand_rw;
        }
      } else if (cand_pim.valid) {
        cand = cand_pim;
      } else if (cand_rw.valid) {
        cand = cand_rw;
      }
    }

    // We have a valid request to serve this cycle
    if (cand.valid) {
      // Rowpolicy *may* upgrade the command to AutoPrecharge version
      m_rowpolicy->try_upgrade_command(*cand.it);

      if (!cand.it->is_stat_updated) {
        update_request_stats(cand.it);
      }

      // Issue command to DRAM device
      m_device.issue_command(cand.it->command, cand.it->addr_vec, m_clk);

      // Notify row policy and plugins of the issued command
      m_rowpolicy->on_issue(*cand.it);
      for (auto* p : m_plugins) {
        p->on_issue(*cand.it);
      }

      // Advance request
      if (cand.it->command == cand.it->final_command) {
        retire_request(cand.it, *cand.buffer);
      } else if (m_device.m_spec->command_meta[cand.it->command].is_opening) {
        promote_to_active(cand.it, *cand.buffer);
      }
    }

    // Post-schedule hooks
    m_rowpolicy->post_schedule();
    for (auto* p : m_plugins) {
      p->post_schedule();
    }
  }

  void retire_request(ReqBuffer::iterator& req_it, ReqBuffer& buffer) {
    if (req_it->type_id == 2 || req_it->type_id == 3) {
      if (&buffer == &m_active_buffer) {
        m_active_per_bank[m_device.get_flat_bank_id(req_it->addr_vec)]--;
      }
      req_it->depart = m_clk + m_device.m_spec->read_latency; // Default for now
      if (req_it->callback) {
        req_it->callback(*req_it);
      }
      s_num_read_reqs_served++; // Hack to make stats show up for now
      buffer.remove(req_it);
    } else {
      ControllerBase::retire_request(req_it, buffer);
    }
  }
};

}  // namespace Ramulator
