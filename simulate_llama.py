import os
import sys
import ramulator

def run_simulation(dram_type="DDR4", trace_path="/mnt/d/antigravity/btp1/llama_decode.trace"):
    print(f"\n==========================================")
    print(f" Running Ramulator2 Simulation: {dram_type}")
    print(f" Trace: {trace_path}")
    print(f"==========================================")
    
    # Configure LoadStoreTrace frontend
    frontend = ramulator.frontend.LoadStoreTrace(
        clock_ratio=4,
        path=trace_path,
    )

    if dram_type == "DDR4":
        dram = ramulator.dram.DDR4(
            org_preset="DDR4_8Gb_x8",
            timing_preset="DDR4_2400R",
            rank=1,
            verbose=False,
        )
        ctrl = ramulator.controller.GenericDDR(
            dram=dram,
            scheduler=ramulator.scheduler.FRFCFS(),
            refresh_manager=ramulator.refresh_manager.AllBank(),
            row_policy=ramulator.row_policy.Open(),
            addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
        )
    elif dram_type == "HBM2":
        dram = ramulator.dram.HBM2(
            org_preset="HBM2_8Gb",
            timing_preset="HBM2_2000Mbps",
            verbose=False,
        )
        ctrl = ramulator.controller.HBM12(
            dram=dram,
            scheduler=ramulator.scheduler.FRFCFS(),
            refresh_manager=ramulator.refresh_manager.AllBank(),
            row_policy=ramulator.row_policy.Open(),
            addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
        )
    else:
        raise ValueError(f"Unsupported DRAM type: {dram_type}")

    mem = ramulator.memory_system.GenericDRAM(
        clock_ratio=3,
        controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
    )

    sim = ramulator.Simulation(frontend, mem)
    sim.run()

    stats = sim.stats
    if stats:
        ctrl_stats = stats["memory_system"]["controller"]
        cycles = ctrl_stats.get('cycles', 0)
        reads = ctrl_stats.get('num_read_reqs', 0)
        writes = ctrl_stats.get('num_write_reqs', 0)
        hits = ctrl_stats.get('row_hits', 0)
        misses = ctrl_stats.get('row_misses', 0)
        conflicts = ctrl_stats.get('row_conflicts', 0)
        avg_lat = ctrl_stats.get('avg_read_latency', 0.0)

        total_reqs = reads + writes
        hit_rate = (hits / total_reqs * 100) if total_reqs > 0 else 0
        miss_rate = (misses / total_reqs * 100) if total_reqs > 0 else 0
        conflict_rate = (conflicts / total_reqs * 100) if total_reqs > 0 else 0

        print(f"\n----- Simulation Results ({dram_type}) -----")
        print(f"Total Controller Cycles: {cycles:,}")
        print(f"Avg Read Latency:        {avg_lat:.2f} cycles")
        print(f"Total Requests Served:   {total_reqs:,} (Reads: {reads:,}, Writes: {writes:,})")
        print(f"Row Buffer Hits:         {hits:,} ({hit_rate:.2f}%)")
        print(f"Row Buffer Misses:       {misses:,} ({miss_rate:.2f}%)")
        print(f"Row Buffer Conflicts:    {conflicts:,} ({conflict_rate:.2f}%)")
        print(f"-----------------------------------------\n")

if __name__ == "__main__":
    run_simulation("DDR4")
    run_simulation("HBM2")