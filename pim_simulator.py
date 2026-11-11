class DRAMBank:
    def __init__(self, bank_id, num_rows, num_srbs):
        self.bank_id = bank_id
        self.num_rows = num_rows
        self.num_srbs = num_srbs
        
        self.primary_row_buffer = None
        self.srbs = [None] * num_srbs
        
        # Event counters
        self.stats = {
            "activates": 0,
            "precharges": 0,
            "rb_to_srb_transfers": 0,
            "pim_mac_ops": 0,
            "srb_to_array_flushes": 0
        }

    def activate(self, row_id):
        if self.primary_row_buffer is not None and self.primary_row_buffer != row_id:
            self.precharge()
            
        if self.primary_row_buffer != row_id:
            self.primary_row_buffer = row_id
            self.stats["activates"] += 1

    def precharge(self):
        self.primary_row_buffer = None
        self.stats["precharges"] += 1

    def copy_to_srb(self, srb_idx):
        if self.primary_row_buffer is None:
            raise ValueError(f"Bank {self.bank_id}: Cannot copy to SRB, PRB is empty.")
        self.srbs[srb_idx] = self.primary_row_buffer
        self.stats["rb_to_srb_transfers"] += 1

    def pim_mac_dot_product(self, srb_a_idx, srb_b_idx, srb_acc_idx):
        if self.srbs[srb_a_idx] is None or self.srbs[srb_b_idx] is None:
            raise ValueError(f"Bank {self.bank_id}: PIM operands not loaded in SRBs.")
        self.stats["pim_mac_ops"] += 1

    def flush_srb(self, srb_idx, row_id):
        self.stats["srb_to_array_flushes"] += 1


class MemoryController:
    def __init__(self, num_banks, num_rows, num_srbs):
        self.banks = [DRAMBank(i, num_rows, num_srbs) for i in range(num_banks)]
        self.cpu_mem_transfers = 0

    def print_stats(self, title):
        total_acts = sum(b.stats["activates"] for b in self.banks)
        total_pres = sum(b.stats["precharges"] for b in self.banks)
        total_pim = sum(b.stats["pim_mac_ops"] for b in self.banks)
        
        print(f"\n--- {title} ---")
        print(f"CPU <-> Mem Transfers : {self.cpu_mem_transfers}")
        print(f"Total Activates       : {total_acts}")
        print(f"Total Precharges      : {total_pres}")
        print(f"Total PIM MAC ops     : {total_pim}")
        print(f"------------------------\n")


def simulate_matmul_pim(M, K, N):
    num_banks = 4
    num_srbs = 4
    num_rows = 65536
    ctrl = MemoryController(num_banks, num_rows, num_srbs)
    
    print(f"Starting PIM-DRAM Simulation: {M}x{K} * {K}x{N}")
    
    for i in range(M):
        for j in range(N):
            bank = ctrl.banks[0]
            
            # 1. Open A[i] and copy to SRB 1
            row_A = i 
            bank.activate(row_A)
            bank.copy_to_srb(1)
            
            # 2. Open B^T[j] and copy to SRB 2
            row_B_T = M + j 
            bank.activate(row_B_T)
            bank.copy_to_srb(2)
            
            # 3. Perform PIM MAC (accumulate into SRB 0)
            bank.pim_mac_dot_product(srb_a_idx=1, srb_b_idx=2, srb_acc_idx=0)
            
            # 4. Flush the result C[i][j] to memory
            row_C = M + N + (i * N + j)
            bank.flush_srb(0, row_C)
            
    # At the end, CPU reads the final C matrix
    ctrl.cpu_mem_transfers += (M * N)

    ctrl.print_stats("PIM-DRAM Stats")


def simulate_matmul_baseline(M, K, N):
    num_banks = 4
    num_srbs = 0 # No SRBs in baseline
    num_rows = 65536
    ctrl = MemoryController(num_banks, num_rows, num_srbs)
    
    print(f"Starting Baseline (Vanilla DRAM) Simulation: {M}x{K} * {K}x{N}")
    
    for i in range(M):
        for j in range(N):
            bank = ctrl.banks[0]
            
            row_A = i 
            row_B_T = M + j 
            row_C = M + N + (i * N + j)

            # CPU reads row A
            bank.activate(row_A)
            ctrl.cpu_mem_transfers += 1 # Transfer row A to CPU
            
            # CPU reads row B^T
            # Since row A was in PRB and we are accessing the same bank,
            # this causes a row buffer conflict. activate() handles precharge automatically.
            bank.activate(row_B_T)
            ctrl.cpu_mem_transfers += 1 # Transfer row B^T to CPU
            
            # CPU computes MAC (no PIM stats incremented)
            
            # CPU writes result to row C
            bank.activate(row_C)
            ctrl.cpu_mem_transfers += 1 # Transfer partial C to Mem
            
    ctrl.print_stats("Baseline Vanilla DRAM Stats")


if __name__ == "__main__":
    M, K, N = 1024, 1024, 1024
    simulate_matmul_baseline(M, K, N)
    simulate_matmul_pim(M, K, N)