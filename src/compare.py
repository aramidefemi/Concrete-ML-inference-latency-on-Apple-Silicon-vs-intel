"""
Compare pilot results across machines and declare the inference server.

Reads all results/pilot_*.json files and prints a side-by-side table.
The machine with the LOWEST fhe_execute_median_ms should be the inference
server; the other becomes the k6 load-testing client.
"""
import os
import json
import glob

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_results():
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "pilot_*.json")))
    results = []
    for p in paths:
        try:
            with open(p) as f:
                results.append(json.load(f))
        except Exception as e:
            print(f"[warn] could not read {p}: {e}")
    return results


def col(s, w):
    s = str(s)
    return s[:w].ljust(w)


def main():
    results = load_results()

    if not results:
        print("No result files found in results/. "
              "Run src/run_pilot.py on each machine first.")
        return

    if len(results) == 1:
        print("Only one result file found. Run the pilot on the second "
              "machine and commit its result before comparing.\n")

    # Header
    w = 22
    print("=" * (w * (len(results) + 1)))
    print("HE PILOT — CROSS-MACHINE COMPARISON")
    print("=" * (w * (len(results) + 1)))

    rows = [
        ("Hostname", lambda r: r["hardware"]["hostname"]),
        ("Architecture", lambda r: r["hardware"]["arch_label"]),
        ("CPU", lambda r: (r["hardware"]["cpu_model"] or "")[:20]),
        ("Cores", lambda r: r["hardware"]["physical_cores"]),
        ("RAM (GB)", lambda r: r["hardware"]["ram_gb"]),
        ("AVX-512", lambda r: r["hardware"]["has_avx512"]),
        ("Data source", lambda r: r["data_source"]),
        ("n_bits", lambda r: r["n_bits"]),
        ("Plaintext median ms", lambda r: r["timings_ms"]["plaintext"]["median_ms"]),
        ("FHE-exec median ms", lambda r: r["fhe_execute_median_ms"]),
        ("Encryption tax (x)", lambda r: r["encryption_tax_x"]),
    ]

    print(col("Metric", w) + "".join(col(f"Machine {i+1}", w)
                                      for i in range(len(results))))
    print("-" * (w * (len(results) + 1)))
    for label, fn in rows:
        line = col(label, w)
        for r in results:
            try:
                line += col(fn(r), w)
            except Exception:
                line += col("-", w)
        print(line)

    # Decision
    print("=" * (w * (len(results) + 1)))
    ranked = sorted(results, key=lambda r: r["fhe_execute_median_ms"])
    winner = ranked[0]
    print("DECISION")
    print("-" * 40)
    print(f"Inference server  -> {winner['hardware']['hostname']} "
          f"({winner['hardware']['arch_label']}), "
          f"{winner['fhe_execute_median_ms']} ms FHE-execute")
    if len(ranked) > 1:
        loser = ranked[1]
        print(f"k6 load client    -> {loser['hardware']['hostname']} "
              f"({loser['hardware']['arch_label']})")
        speedup = round(loser["fhe_execute_median_ms"] /
                        winner["fhe_execute_median_ms"], 2)
        print(f"\nThe inference server is {speedup}x faster on FHE-execute.")
        print("Note this cross-architecture difference in your methodology "
              "chapter — it is a small reportable finding on HE portability.")


if __name__ == "__main__":
    main()
