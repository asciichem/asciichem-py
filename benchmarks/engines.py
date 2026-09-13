"""Cross-engine parsing benchmark (shared workload; see
asciichem-ruby benchmarks/README.md). Run: python benchmarks/engines.py"""
import sys
import time

sys.path.insert(0, "src")

from asciichem import parse_text

WORKLOAD = ["H_2O", "Ca^2+", "SO_4^2-", "(R)-CH_3CH(OH)COOH",
            "2H_2 + O_2 -> 2H_2O", "N_2 + 3H_2 <=>[Fe][400C] 2NH_3",
            "C1-C-C-C-C-C1", "CH_3-CH_2-OH",
            '^14C @name("carbon-14") @cas("14104-86-4")',
            "A ->[heat] B ->[cool] C"]

for s in WORKLOAD:
    parse_text(s)  # warm-up + correctness gate

N = 300
t0 = time.perf_counter()
for _ in range(N):
    for s in WORKLOAD:
        parse_text(s)
ms = (time.perf_counter() - t0) * 1000
print(f"rd: {N} batches of 10 in {ms:.1f} ms")
print(f"batch: {ms / N:.2f} ms; per input: {ms / N / 10 * 1000:.0f} µs")
