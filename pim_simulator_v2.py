class DRAMBank:
    def __init__(self, bank_id):
        self.bank_id = bank_id
        # Event counters
        self.stats = {
            "activates": 0,
            "precharges": 0,
            "pim_mac_ops": 0,
            "inter_bank_transfers_in": 0,
            "inter_bank_transfers_out": 0
        }

    def record_activate(self, count=1):
        self.stats["activates"] += count
        self.stats["precharges"] += count  # Assuming every activate eventually needs a precharge

    def record_pim_mac(self, count=1):
        self.stats["pim_mac_ops"] += count

    def record_transfer_in(self, count=1):
        self.stats["inter_bank_transfers_in"] += count
        
    def record_transfer_out(self, count=1):
        self.stats["inter_bank_transfers_out"] += count

class MemoryController:
    def __init__(self, num_banks):
        self.banks = [DRAMBank(i) for i in range(num_banks)]
        self.cpu_mem_transfers = 0

    def print_stats(self, title):
        total_acts = sum(b.stats["activates"] for b in self.banks)
        total_pres = sum(b.stats["precharges"] for b in self.banks)
        total_pim = sum(b.stats["pim_mac_ops"] for b in self.banks)
        total_ib_in = sum(b.stats["inter_bank_transfers_in"] for b in self.banks)
        total_ib_out = sum(b.stats["inter_bank_transfers_out"] for b in self.banks)
        
        print(f"\n=============================================")
        print(f" {title}")
        print(f"=============================================")
        print(f"CPU <-> Mem Transfers   : {self.cpu_mem_transfers:,}")
        print(f"Inter-Bank Transfers    : {total_ib_in:,} (Rows Moved)")
        print(f"Total Activates         : {total_acts:,}")
        print(f"Total Precharges        : {total_pres:,}")
        print(f"Total PIM MAC ops       : {total_pim:,}")
        print(f"=============================================\n")

def simulate_matmul_pim(M, K, N, elem_bytes=4, row_size_bytes=8192, num_banks=16):
    ctrl = MemoryController(num_banks)
    
    elems_per_row = row_size_bytes // elem_bytes
    dram_rows_per_matrix_row = (K + elems_per_row - 1) // elems_per_row
    
    print(f"Simulating PIM-DRAM Matrix Multiplication")
    print(f"Matrix Size: {M}x{K} * {K}x{N} ({elem_bytes} bytes/element)")
    print(f"Elements per DRAM Row: {elems_per_row}")
    print(f"DRAM Rows required per Matrix Row: {dram_rows_per_matrix_row}")
    print(f"Number of Banks: {num_banks}")
    
    # 1. Data Mapping
    # Matrix B^T is interleaved across banks. 
    # Each bank gets N / num_banks rows of B^T.
    rows_B_per_bank = N // num_banks
    
    # Matrix A is stored starting in Bank 0, we'll assume it needs to be broadcasted.
    
    # 2. Execution Flow (Streaming Activations to Stationary Weights)
    # For every row in A:
    for i in range(M):
        # We process the row in chunks that fit into a DRAM row
        for p in range(dram_rows_per_matrix_row):
            # A[i][p] is read from its source bank (say Bank 0)
            src_bank = ctrl.banks[0]
            src_bank.record_activate(1) 
            src_bank.record_transfer_out(num_banks - 1) # Broadcast to other banks
            
            # The chunk A[i][p] is broadcast to all banks.
            for bank_idx in range(num_banks):
                bank = ctrl.banks[bank_idx]
                if bank_idx != 0:
                    bank.record_transfer_in(1)
                
                # The bank caches A[i][p] in its SRB.
                # Now, it multiplies this chunk with all its local rows of B^T
                
                # To do this, the bank must activate each of its local B^T chunks
                # and do a PIM MAC operation.
                
                # Number of local B^T rows is rows_B_per_bank.
                # Each needs 1 activate for chunk p.
                bank.record_activate(rows_B_per_bank)
                
                # Each MAC processes the entire DRAM row
                bank.record_pim_mac(rows_B_per_bank)
                
    # At the end, CPU reads the final C matrix
    # C is M x N. Each row of C has N elements.
    dram_rows_for_C = M * ((N + elems_per_row - 1) // elems_per_row)
    ctrl.cpu_mem_transfers += dram_rows_for_C

    ctrl.print_stats("PIM-DRAM Stats (Distributed Banks & Broadcast)")

def simulate_matmul_baseline(M, K, N, elem_bytes=4, row_size_bytes=8192, num_banks=16):
    ctrl = MemoryController(num_banks)
    elems_per_row = row_size_bytes // elem_bytes
    dram_rows_per_matrix_row = (K + elems_per_row - 1) // elems_per_row
    
    print(f"Simulating Baseline Vanilla DRAM")
    
    # Baseline assumes CPU must fetch every chunk of A and B^T for the dot products,
    # assuming a realistic Last-Level Cache (LLC) that caches A, but streams B^T.
    # To compute C, we stream B^T once if A fits in cache, but A is 8192x8192 (256MB),
    # which exceeds most LLCs. Let's assume standard blocked matrix multiplication memory traffic.
    # A naive bound: Fetching A once (256MB), fetching B^T once for every row of A?
    # No, with blocking, we fetch A and B multiple times. 
    # For a purely theoretical event counting without cache:
    # CPU reads A[i] (dram_rows_per_matrix_row) and B^T[j] (dram_rows_per_matrix_row) for every C[i][j].
    
    # Let's count DRAM rows fetched:
    # Each C[i][j] needs A[i] and B^T[j].
    total_C_elements = M * N
    
    # If NO cache (worst case):
    # CPU requests A[i] and B^T[j] for all elements
    reads_A = M * N * dram_rows_per_matrix_row
    reads_B = M * N * dram_rows_per_matrix_row
    writes_C = M * ((N + elems_per_row - 1) // elems_per_row)
    
    ctrl.cpu_mem_transfers = reads_A + reads_B + writes_C
    
    # Distribute the activates across banks evenly
    total_activates = reads_A + reads_B + writes_C
    for bank in ctrl.banks:
        bank.record_activate(total_activates // num_banks)
        
    ctrl.print_stats("Baseline Vanilla DRAM Stats (No Cache)")

if __name__ == "__main__":
    M, K, N = 8192, 8192, 8192
    simulate_matmul_baseline(M, K, N)
    simulate_matmul_pim(M, K, N)