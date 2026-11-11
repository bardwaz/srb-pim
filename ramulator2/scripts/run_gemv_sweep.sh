#!/bin/bash
# ==============================================================================
# Cycle-Accurate PIM GEMV Sweep (Ramulator 2.1)
# BTP Presentation Benchmark: N in [1024, 2048, 4096, 8192]
# DRAM Row Size: 8 KB (32 tiles / row) | Bank SRB Capacity: 8 rows (64 KB / bank)
# ==============================================================================

export PYTHONPATH=/mnt/d/antigravity/btp1/ramulator2/python
cd /mnt/d/antigravity/btp1/ramulator2

echo "=========================================================================================="
echo " Cycle-Accurate PIM GEMV Evaluation (128 Banks, 8KB DRAM/SRB Rows, 64KB SRB/Bank)"
echo "=========================================================================================="
printf "%-8s | %-10s | %-16s | %-10s | %-16s | %-10s\n" "Size (N)" "CPU (ms)" "SIMD-8 (ms)" "Speedup" "Vec-4 (ms)" "Speedup"
echo "------------------------------------------------------------------------------------------"

declare -A cpu_t=( ["1024"]="0.775" ["2048"]="3.312" ["4096"]="12.782" ["8192"]="52.821" )

for N in 1024 2048 4096 8192; do
    # Run SIMD (8 cycles / MAC)
    t_simd=$(python3 examples/pim_gemv_streaming.py $N autonomous 8 2>&1 | awk '/Execution time:/ {print $3}')
    
    # Run Vector (4 cycles / MAC)
    t_vec=$(python3 examples/pim_gemv_streaming.py $N autonomous 4 2>&1 | awk '/Execution time:/ {print $3}')
    
    cpu=${cpu_t[$N]}
    
    sp_simd=$(awk -v c="$cpu" -v s="$t_simd" 'BEGIN {if (s>0) printf "%.1fx", c/s; else print "N/A"}')
    sp_vec=$(awk -v c="$cpu" -v v="$t_vec" 'BEGIN {if (v>0) printf "%.1fx", c/v; else print "N/A"}')
    
    printf "%-8s | %-10s | %-16s | %-10s | %-16s | %-10s\n" "$N" "$cpu" "$t_simd" "$sp_simd" "$t_vec" "$sp_vec"
done

echo "=========================================================================================="
