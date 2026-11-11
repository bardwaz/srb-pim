class DRAMBank:
    def __init__(self, bank_id):
        self.bank_id = bank_id
        # Event counters
        self.stats = {
            "activates": 0,
            "precharges": 0,
            "pim_vector_mac_ops": 0,
        }

    def record_activate(self, count=1):
        self.stats["activates"] += count
        self.stats["precharges"] += count

    def record_pim_mac(self, count=1):
        self.stats["pim_vector_mac_ops"] += count


class MemoryController:
    def __init__(self, num_banks):
        self.banks = [DRAMBank(i) for i in range(num_banks)]
        self.cpu_mem_transfers = 0

    def print_stats(self):
        total_acts = sum(b.stats["activates"] for b in self.banks)
        total_pres = sum(b.stats["precharges"] for b in self.banks)
        total_pim = sum(b.stats["pim_vector_mac_ops"] for b in self.banks)
        
        print(f"\n=======================================================")
        print(f" PIM-DRAM Stats (Outer-Product 2D Grid Mapping)")
        print(f"=======================================================")
        print(f"CPU <-> Mem Transfers     : {self.cpu_mem_transfers:,} (Reading final C matrix)")
        print(f"Total Activates           : {total_acts:,}")
        print(f"Total Precharges          : {total_pres:,}")
        print(f"Total Vector MAC ops      : {total_pim:,}")
        print(f"  *(Each Vector MAC processes 2048 scalar multiplications)*")
        print(f"  *(Total equivalent scalar MACs = {total_pim * 2048:,})*")
        print(f"=======================================================\n")


def simulate_matmul_pim_v3(M, K, N, elem_bytes=4, row_size_bytes=8192):
    # Mapping Strategy:
    # C is divided into 16 square blocks in a 4x4 grid.
    grid_dim = 4
    num_banks = grid_dim * grid_dim
    ctrl = MemoryController(num_banks)
    
    elems_per_row = row_size_bytes // elem_bytes
    dram_rows_per_k_dim = (K + elems_per_row - 1) // elems_per_row
    
    print(f"Simulating PIM-DRAM Matrix Multiplication (V3 - Outer Product)")
    print(f"Matrix Size: {M}x{K} * {K}x{N} ({elem_bytes} bytes/element)")
    print(f"C Matrix partitioned into {grid_dim}x{grid_dim} blocks.")
    print(f"Number of Banks: {num_banks}")
    
    # Each bank computes a block of C of size 2048 x 2048
    rows_A_per_bank = M // grid_dim
    cols_B_per_bank = N // grid_dim
    
    print(f"Each bank processes a {rows_A_per_bank} x {cols_B_per_bank} block of the output.")
    print(f"This requires an A block of {rows_A_per_bank}x{K} and a B block of {K}x{cols_B_per_bank}.")
    
    # We use OUTER PRODUCT formulation.
    # To compute a row of C (2048 elements), we take scalars from A and multiply them
    # across a full row of B. This eliminates the need for an adder-tree in the DRAM!
    # B is stored normally (NOT transposed).
    
    for bank_idx in range(num_banks):
        bank = ctrl.banks[bank_idx]
        
        # The bank computes an output block row by row.
        for i in range(rows_A_per_bank):
            
            # For a single row of C, we need the corresponding row of A (length K=8192).
            # This row of A spans 4 DRAM rows. We load it chunk by chunk into SRB1.
            for p in range(dram_rows_per_k_dim):
                # 1. Activate the p-th chunk of A into SRB1
                bank.record_activate(1)
                
                # This chunk has 2048 scalar elements.
                # We iterate through each scalar, broadcasting it across a row of B.
                for k_elem in range(elems_per_row):
                    # 2. Activate the corresponding row of B into SRB2
                    # (This is 1 DRAM row since B block width is 2048)
                    bank.record_activate(1)
                    
                    # 3. Perform Vector-Scalar MAC: SRB0 += SRB1[k_elem] * SRB2
                    bank.record_pim_mac(1)
                    
    # At the end, CPU reads the final C matrix from all banks
    dram_rows_for_C = M * ((N + elems_per_row - 1) // elems_per_row)
    ctrl.cpu_mem_transfers += dram_rows_for_C

    ctrl.print_stats()


if __name__ == "__main__":
    M, K, N = 8192, 8192, 8192
    simulate_matmul_pim_v3(M, K, N)