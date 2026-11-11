# SRB-PIM

Cycle-accurate Ramulator 2.1 simulation and evaluation codebase for a Decoupled Secondary Row Buffer (SRB) Processing-in-Memory architecture targeting matrix-vector (GEMV) and matrix-matrix (GEMM) kernels.

## Structure

- `ieee_btp_report.tex`: IEEE conference report source.
- `ramulator2/`: Modified Ramulator 2.1 simulator with DDR4 SRB-PIM support.
  - `examples/pim_gemv_streaming.py`: Cycle-accurate GEMV simulation driver.
  - `examples/pim_gemm_streaming.py`: GEMM simulation driver.
  - `scripts/`: Single-threaded (`-O2`) and AVX2+FMA OpenMP CPU baselines for GEMV and GEMM.
- `pim_simulator*.py`, `gemm.cpp`, `cg_log*.txt`: Standalone analytical prototypes and Cachegrind profiling logs.

## Ramulator 2.1 Modifications (under `ramulator2/`)

- DRAM spec and timing presets (`python/ramulator/dram/ddr4_pim.py`, generated `src/ramulator/dram/impl/DDR4_PIM.cpp`): adds `PIM_ACT_SRB`, `PIM_MAC`, `PIM_FLUSH`, timing parameters `nSRB_COPY`, `nPIM_MAC`, `nSRB_FLUSH`, and presets `DDR4_2400R_PIM` (32 cyc/MAC), `DDR4_2400R_PIM_SIMD` (8 cyc/MAC), and `DDR4_2400R_PIM_VEC` (4 cyc/MAC).
- Per-bank SRB state (`src/ramulator/dram/node.h`, `src/ramulator/dram/commands/PIM_*.h`): tracks an 8-row per-bank FIFO SRAM SRB (`m_srb_rows`, 64 KB at 8 KB/row); `PIM_ACT_SRB` loads a row and auto-precharges primary bitlines.
- Controller and streaming frontends (`src/ramulator/controller/impl/pim_controller.cpp`, `src/ramulator/frontend/impl/memory_trace/pim_gemv_frontend.cpp`, `pim_gemm_frontend.cpp`): adds a PIM request queue (`m_pim_buffer`) and O(1)-memory streaming GEMV and outer-product GEMM request generators.

## Build and Run

Build Ramulator 2.1 (Linux/WSL, CMake >= 3.14, C++20, Python >= 3.10):

```bash
cd ramulator2
./build.sh
```

Run PIM simulations (`args: <N> [autonomous|shared] [32|8|4]`):

```bash
PYTHONPATH=ramulator2/python python3 ramulator2/examples/pim_gemv_streaming.py 8192 autonomous 8
PYTHONPATH=ramulator2/python python3 ramulator2/examples/pim_gemm_streaming.py 1024 autonomous 8
```

Compile CPU baselines (from `ramulator2/scripts/`):

```bash
g++ -O2 cpu_baseline_gemv.cpp -o cpu_gemv_naive
g++ -O3 -march=native -mavx2 -mfma -fopenmp cpu_baseline_gemv_optimized.cpp -o cpu_gemv_opt
g++ -O2 cpu_baseline_gemm.cpp -o cpu_gemm_naive
g++ -O3 -march=native -mavx2 -mfma -fopenmp cpu_baseline_gemm_optimized.cpp -o cpu_gemm_opt
```
