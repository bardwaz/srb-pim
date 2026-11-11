import math

class Config:
    # Hardware constraints parameterized
    ELEM_BYTES = 4               
    ROW_SIZE_BYTES = 8192        
    
    # Timing parameters
    CLOCK_FREQ_MHZ = 1000        
    TACT_NS = 15                 
    TPRE_NS = 15                 
    
    # Compute parameters
    SIMD_WIDTH = 128               
    PIPELINED_SRB = True         # Use SRBs to Double-Buffer memory fetches and compute!
    
    # Baseline CPU constraints
    CPU_MEM_BW_GBPS = 50.0       

    @classmethod
    def get_tact_cycles(cls):
        return math.ceil(cls.TACT_NS * (cls.CLOCK_FREQ_MHZ / 1000.0))
        
    @classmethod
    def get_tpre_cycles(cls):
        return math.ceil(cls.TPRE_NS * (cls.CLOCK_FREQ_MHZ / 1000.0))


class DRAMBank:
    def __init__(self, bank_id):
        self.bank_id = bank_id
        self.current_cycle = 0
        self.stats = {
            "activates": 0,
            "precharges": 0,
            "pim_vector_mac_ops": 0,
        }

    def add_bulk_operations(self, a_chunks, b_activates, compute_elements_per_mac):
        self.stats["activates"] += (a_chunks + b_activates)
        self.stats["precharges"] += (a_chunks + b_activates)
        self.stats["pim_vector_mac_ops"] += b_activates
        
        # A chunk fetches (not pipelined, since we wait for A before doing B sweep)
        tact_cycles_a = Config.get_tact_cycles() * a_chunks
        tpre_cycles_a = Config.get_tpre_cycles() * a_chunks
        self.current_cycle += (tact_cycles_a + tpre_cycles_a)
        
        # B sweeps (PIPELINED with SRBs)
        mem_cycles_per_b = Config.get_tact_cycles() + Config.get_tpre_cycles()
        compute_cycles_per_b = math.ceil(compute_elements_per_mac / Config.SIMD_WIDTH)
        
        if Config.PIPELINED_SRB:
            # Double-buffering using Multiple Row Buffers!
            # The time taken is the maximum of the memory array latency or the ALU compute latency.
            effective_cycles_per_b = max(mem_cycles_per_b, compute_cycles_per_b)
        else:
            effective_cycles_per_b = mem_cycles_per_b + compute_cycles_per_b
            
        self.current_cycle += (effective_cycles_per_b * b_activates)


class MemoryController:
    def __init__(self, num_banks):
        self.banks = [DRAMBank(i) for i in range(num_banks)]

    def print_stats(self, M, K, N):
        max_cycles = max(b.current_cycle for b in self.banks)
        pim_time_ms = (max_cycles / (Config.CLOCK_FREQ_MHZ * 1e6)) * 1000
        baseline_time_ms = (47.1 / Config.CPU_MEM_BW_GBPS) * 1000
        
        print(f"\n=======================================================")
        print(f" PIM-DRAM Stats (Pipelined MRB Architecture)")
        print(f"=======================================================")
        print(f"Hardware Config:")
        print(f"  Architecture: 16 Banks (4x4 Grid, Standard DDR4 footprint)")
        print(f"  Clock: {Config.CLOCK_FREQ_MHZ} MHz, SIMD Width: {Config.SIMD_WIDTH} elements/cycle")
        print(f"  SRB Pipelining (Double Buffering): {'ENABLED' if Config.PIPELINED_SRB else 'DISABLED'}")
        print(f"-------------------------------------------------------")
        print(f"Baseline CPU Execution Time  : {baseline_time_ms:,.2f} ms")
        print(f"PIM-DRAM Execution Time      : {pim_time_ms:,.2f} ms")
        print(f"Speedup Factor               : {baseline_time_ms / pim_time_ms:.2f}x")
        print(f"=======================================================\n")


def simulate_matmul_pipelined_pim(M, K, N):
    grid_dim = 4
    num_banks = grid_dim * grid_dim
    ctrl = MemoryController(num_banks)
    
    elems_per_row = Config.ROW_SIZE_BYTES // Config.ELEM_BYTES
    dram_rows_per_k_dim = (K + elems_per_row - 1) // elems_per_row
    
    rows_A_per_bank = M // grid_dim
    
    for bank_idx in range(num_banks):
        bank = ctrl.banks[bank_idx]
        
        total_a_chunks = rows_A_per_bank * dram_rows_per_k_dim
        total_b_activates = total_a_chunks * elems_per_row
        
        bank.add_bulk_operations(
            a_chunks=total_a_chunks, 
            b_activates=total_b_activates, 
            compute_elements_per_mac=elems_per_row
        )

    ctrl.print_stats(M, K, N)


if __name__ == "__main__":
    simulate_matmul_pipelined_pim(8192, 8192, 8192)