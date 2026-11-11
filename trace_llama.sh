#!/bin/bash
set -e

echo "Starting Valgrind memory trace collection..."
valgrind --tool=lackey --trace-mem=yes \
    /mnt/d/BTP/week1/llama.cpp/build-wsl/bin/llama-bench \
    -m /mnt/d/BTP/week1/llama.cpp/build/bin/model.gguf \
    -p 0 -n 1 -r 1 -t 1 2>&1 \
    | awk '
        / L / { split(\, a, ","); print "LD 0x" a[1] }
        / S / { split(\, a, ","); print "ST 0x" a[1] }
        / M / { split(\, a, ","); print "LD 0x" a[1]; print "ST 0x" a[1] }
    ' | head -n 500000 > /mnt/d/antigravity/btp1/llama_decode.trace

echo "Trace generation completed! Lines generated:"
wc -l /mnt/d/antigravity/btp1/llama_decode.trace
head -n 10 /mnt/d/antigravity/btp1/llama_decode.trace