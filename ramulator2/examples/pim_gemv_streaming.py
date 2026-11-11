import ramulator
import sys

# Matrix-Vector Multiplication: y[M] = A[M x N] * x[N]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
M = N
mode = sys.argv[2] if len(sys.argv) > 2 else "autonomous"  # "autonomous" or "shared"
mac_cycles = int(sys.argv[3]) if len(sys.argv) > 3 else 32  # 32 (scalar), 8 (SIMD), 4 (vector)

# Standardized 8 KB DRAM & SRB row size: 32 tiles * 256 bytes = 8192 bytes
TILES_PER_ROW = 32

if N > 8192:
    print(f"Error: Matrix dimension N={N} exceeds maximum supported BTP cap of 8192.")
    sys.exit(1)

print(f"Running PIM GEMV: y[{M}] = A[{M}x{N}] * x[{N}]")
print(f"  Mode: {mode.upper()}, MAC latency: {mac_cycles} cyc, Row Buffer: 8 KB ({TILES_PER_ROW} tiles)")

num_banks = 128

if mode == "autonomous":
    num_channels = 128  # 1 channel per bank (autonomous DPU model, zero bus contention)
else:
    num_channels = 8    # 8 channels x 16 banks = 128 banks (shared command bus)

# Select DRAM timing preset matching ALU datapath width
timing_presets = {
    32: "DDR4_2400R_PIM",       # Scalar ALU
    8:  "DDR4_2400R_PIM_SIMD",  # 256-bit SIMD ALU
    4:  "DDR4_2400R_PIM_VEC",   # 512-bit Vector ALU
}
timing_preset = timing_presets.get(mac_cycles, "DDR4_2400R_PIM")

controllers = []
for i in range(num_channels):
    ctrl = ramulator.controller.PIMController(
        dram=ramulator.dram.DDR4_PIM(
            org_preset="DDR4_8Gb_x8",
            timing_preset=timing_preset,
            rank=1,
            verbose=False,
        ),
        scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=ramulator.refresh_manager.NoRefresh(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.PassThroughAddrMapper(),
    )
    controllers.append(ctrl)

mem = ramulator.memory_system.GenericDRAM(
    clock_ratio=3,
    controllers=controllers,
    channel_mapper=ramulator.channel_mapper.PassThroughChannelMapper(),
)

frontend = ramulator.frontend.PIMGemvFrontend(
    clock_ratio=3,
    M=M,
    N=N,
    num_banks=num_banks,
    num_channels=num_channels,
    banks_per_bg=4,
    pim_mac_cycles=mac_cycles,
    tiles_per_row=TILES_PER_ROW,
)

sim = ramulator.Simulation(frontend, mem)
sim.run()

stats = sim.stats
if stats:
    ctrl_stats_list = stats["memory_system"]["controller"]
    ctrl_stats = ctrl_stats_list[0] if isinstance(ctrl_stats_list, list) else ctrl_stats_list
    tCK_ps = 833  # DDR4-2400 (1.2 GHz)
    cycles = ctrl_stats["cycles"]
    time_ns = cycles * tCK_ps / 1000
    time_ms = time_ns / 1e6

    print(f"\n{'='*60}")
    print(f" PIM-DRAM Cycle-Accurate GEMV Results (Ramulator 2.1)")
    print(f" Matrix: {M}x{N}, Mode: {mode.upper()} ({num_channels} Channels, {num_banks} Banks)")
    print(f" ALU: nPIM_MAC={mac_cycles} cyc, Row Buffer: 8 KB")
    print(f"{'='*60}")
    print(f"Total cycles:          {cycles:,}")
    print(f"Execution time:        {time_ms:.4f} ms")
    print(f"{'='*60}")
