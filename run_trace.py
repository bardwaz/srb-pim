import subprocess
import sys

valgrind_cmd = [
    "valgrind", "--tool=lackey", "--trace-mem=yes",
    "/mnt/d/BTP/week1/llama.cpp/build-wsl/bin/llama-bench",
    "-m", "/mnt/d/BTP/week1/llama.cpp/build/bin/model.gguf",
    "-p", "0", "-n", "1", "-r", "1", "-t", "1"
]

output_file = "/mnt/d/antigravity/btp1/llama_decode.trace"
max_lines = 500000

print("Running Valgrind to capture memory trace...")
proc = subprocess.Popen(valgrind_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

count = 0
with open(output_file, "w") as out:
    for line in proc.stdout:
        line = line.strip()
        parts = line.split()
        if len(parts) >= 2:
            op = parts[0]
            if op in ("L", "S", "M"):
                addr = parts[1].split(",")[0]
                if op == "L":
                    out.write(f"LD 0x{addr}\n")
                    count += 1
                elif op == "S":
                    out.write(f"ST 0x{addr}\n")
                    count += 1
                elif op == "M":
                    out.write(f"LD 0x{addr}\n")
                    out.write(f"ST 0x{addr}\n")
                    count += 2
                
                if count >= max_lines:
                    print(f"Captured {count} memory operations. Terminating tracer...")
                    proc.terminate()
                    break

print(f"Done! Generated {count} trace entries in {output_file}")