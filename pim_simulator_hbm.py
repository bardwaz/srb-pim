import math

class Config:
    ELEM_BYTES = 4               
    ROW_SIZE_BYTES = 8192        
    CLOCK_FREQ_MHZ = 1000        
    TACT_NS = 15                 
    TPRE_NS = 15                 
    SIMD_WIDTH = 128               
    HOST_MEM_BW_GBPS = 460.0       

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

    def add_bulk_operations(self, activates, precharges, vector_macs, compute_elements_per_mac):
        self.stats["activates"] += activates
        self.stats["precharges"] += precharges
        self.stats["pim_vector_mac_ops"] += vector_macs
        
        tact_cycles = Config.get_tact_cycles() * activates
        tpre_cycles = Config.get_tpre_cycles() * precharges
        compute_cycles = math.ceil(compute_elements_per_mac / Config.SIMD_WIDTH) * vector_macs
        
        self.current_cycle += (tact_cycles + tpre_cycles + compute_cycles)


class MemoryController:
    def __init__(self, num_banks):
        self.banks = [DRAMBank(i) for i in range(num_banks)]

    def print_stats(self, M, K, N):
        max_cycles = max(b.current_cycle for b in self.banks)
        pim_time_ms = (max_cycles / (Config.CLOCK_FREQ_MHZ * 1e6)) * 1000
        baseline_time_ms = (47.1 / Config.HOST_MEM_BW_GBPS) * 1000
        
        print(f"\n=======================================================")
        print(f" HBM-PIM Stats (Cycle-Accurate Analytical Model)")
        print(f"=======================================================")
        print(f"Hardware Config:")
        print(f"  Architecture: 256 Banks (16x16 Grid)")
        print(f"  Clock: {Config.CLOCK_FREQ_MHZ} MHz, SIMD Width: {Config.SIMD_WIDTH} elements/cycle")
        print(f"-------------------------------------------------------")
        print(f"Baseline Host (HBM2) Execution Time (Theoretical) : {baseline_time_ms:,.2f} ms")
        print(f"HBM-PIM Execution Time                            : {pim_time_ms:,.2f} ms")
        print(f"Speedup Factor over Theoretical Peak              : {baseline_time_ms / pim_time_ms:.2f}x")
        print(f"=======================================================\n")


def simulate_matmul_hbm_pim(M, K, N):
    grid_dim = 16
    num_banks = grid_dim * grid_dim
    ctrl = MemoryController(num_banks)
    
    elems_per_row = Config.ROW_SIZE_BYTES // Config.ELEM_BYTES
    dram_rows_per_k_dim = (K + elems_per_row - 1) // elems_per_row
    
    rows_A_per_bank = M // grid_dim
    cols_B_per_bank = N // grid_dim  # In 16x16, this is 512 elements
    
    for bank_idx in range(num_banks):
        bank = ctrl.banks[bank_idx]
        
        total_a_chunks = rows_A_per_bank * dram_rows_per_k_dim
        total_b_activates = total_a_chunks * elems_per_row
        
        bank.add_bulk_operations(activates=total_a_chunks, precharges=total_a_chunks, vector_macs=0, compute_elements_per_mac=0)
        # Fix: The B block width is only cols_B_per_bank (512), not the full 2048 elements!
        bank.add_bulk_operations(activates=total_b_activates, precharges=total_b_activates, vector_macs=total_b_activates, compute_elements_per_mac=cols_B_per_bank)

    ctrl.print_stats(M, K, N)


if __name__ == "__main__":
    simulate_matmul_hbm_pim(8192, 8192, 8192)