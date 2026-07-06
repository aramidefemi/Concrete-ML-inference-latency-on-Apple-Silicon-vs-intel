"""
HE Inference Pilot
==================
Trains a small feedforward NN on the ULB fraud dataset, compiles it to FHE with
Concrete ML, and times plaintext vs FHE-simulate vs FHE-execute single-request
inference. Writes a hardware-tagged result file.

Run on both machines, then use compare.py to decide the inference host.
"""
import os
import json
import time
import statistics
from datetime import datetime, timezone

import numpy as np

from hwinfo import get_hw_info

# ---- Config -------------------------------------------------------------
DATA_PATH = os.environ.get("PILOT_DATA", "creditcard.csv")
N_TIMING_RUNS = int(os.environ.get("PILOT_RUNS", "30"))   # timed FHE-execute runs
N_FEATURES = 30                                           # ULB has 30 features
RANDOM_SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# n_bits controls the FHE quantization precision/speed tradeoff.
# Lower = faster but less precise. 6 is a sensible pilot default.
N_BITS = int(os.environ.get("PILOT_NBITS", "6"))


def load_data():
    """Load ULB dataset, or generate a synthetic fallback so the pipeline
    can be smoke-tested without the real file."""
    try:
        import pandas as pd
        df = pd.read_csv(DATA_PATH)
        # Standard ULB columns: Time, V1..V28, Amount, Class
        y = df["Class"].values
        X = df.drop(columns=["Class"]).values.astype(np.float32)
        print(f"[data] Loaded real ULB dataset: {X.shape[0]} rows, {X.shape[1]} features")
        source = "ULB_real"
    except FileNotFoundError:
        print(f"[data] {DATA_PATH} not found — generating synthetic fallback "
              f"(pipeline smoke test only, NOT for real results)")
        rng = np.random.default_rng(RANDOM_SEED)
        n = 5000
        X = rng.standard_normal((n, N_FEATURES)).astype(np.float32)
        # crude imbalanced target
        logits = X[:, 0] * 1.5 - X[:, 1] * 0.8 + rng.standard_normal(n) * 0.5
        y = (logits > 3.1).astype(int)  # ~1% positive (smoke test only)
        source = "synthetic_fallback"

    return X, y, source


def preprocess(X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    return X_train, X_test, y_train, y_test


def build_and_compile(X_train, y_train):
    """Train a small NN classifier and compile to FHE using Concrete ML."""
    from concrete.ml.sklearn import NeuralNetClassifier
    import torch

    # Small feedforward net — shallow to keep multiplicative depth low for HE.
    # 2 layers (not 3): 3-layer + 30 features exceeds TFHE parameter search on real ULB data.
    model = NeuralNetClassifier(
        module__n_layers=2,
        module__n_w_bits=N_BITS,
        module__n_a_bits=N_BITS,
        module__n_hidden_neurons_multiplier=1,
        max_epochs=20,
        batch_size=64,
        lr=0.01,
        verbose=0,
    )

    print("[train] Training feedforward classifier (plaintext)...")
    t0 = time.perf_counter()
    model.fit(X_train, y_train.astype(np.int64))
    train_s = time.perf_counter() - t0
    print(f"[train] Done in {train_s:.1f}s")

    print(f"[compile] Compiling to FHE (n_bits={N_BITS})...")
    t0 = time.perf_counter()
    # Calibration subset is enough for range estimation and much faster than full X_train.
    cal = X_train[: min(100, len(X_train))]
    model.compile(cal)
    compile_s = time.perf_counter() - t0
    print(f"[compile] Done in {compile_s:.1f}s")

    return model, train_s, compile_s


def time_inference(model, X_test):
    """Time plaintext, FHE-simulate, and FHE-execute single-request inference."""
    sample = X_test[0:1]  # single request

    # 1) plaintext
    plain_times = []
    for _ in range(N_TIMING_RUNS):
        t0 = time.perf_counter()
        _ = model.predict(sample, fhe="disable")
        plain_times.append((time.perf_counter() - t0) * 1000)

    # 2) FHE simulate (correctness path, cheap)
    sim_times = []
    for _ in range(min(N_TIMING_RUNS, 10)):
        t0 = time.perf_counter()
        _ = model.predict(sample, fhe="simulate")
        sim_times.append((time.perf_counter() - t0) * 1000)

    # 3) FHE execute (real encrypted inference — the decisive number)
    # Fewer runs since each is expensive.
    exec_runs = max(5, N_TIMING_RUNS // 3)
    exec_times = []
    print(f"[time] Running {exec_runs} real FHE-execute inferences "
          f"(this is the slow part)...")
    for i in range(exec_runs):
        t0 = time.perf_counter()
        _ = model.predict(sample, fhe="execute")
        dt = (time.perf_counter() - t0) * 1000
        exec_times.append(dt)
        print(f"  run {i+1}/{exec_runs}: {dt:.1f} ms")

    def summ(xs):
        return {
            "median_ms": round(statistics.median(xs), 3),
            "mean_ms": round(statistics.mean(xs), 3),
            "min_ms": round(min(xs), 3),
            "max_ms": round(max(xs), 3),
            "runs": len(xs),
        }

    return {
        "plaintext": summ(plain_times),
        "fhe_simulate": summ(sim_times),
        "fhe_execute": summ(exec_times),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    hw = get_hw_info()

    print("=" * 60)
    print("HE INFERENCE PILOT")
    print("=" * 60)
    print(f"Host: {hw['hostname']} | Arch: {hw['arch_label']} | "
          f"CPU: {hw['cpu_model']}")
    print(f"Cores: {hw['physical_cores']} | RAM: {hw['ram_gb']}GB | "
          f"AVX512: {hw['has_avx512']}")
    print("=" * 60)

    X, y, source = load_data()
    X_train, X_test, y_train, y_test = preprocess(X, y)
    model, train_s, compile_s = build_and_compile(X_train, y_train)
    timings = time_inference(model, X_test)

    # Encryption tax at single request (execute median vs plaintext median)
    enc_tax = None
    if timings["plaintext"]["median_ms"] > 0:
        enc_tax = round(
            timings["fhe_execute"]["median_ms"] /
            timings["plaintext"]["median_ms"], 1
        )

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": source,
        "n_bits": N_BITS,
        "hardware": hw,
        "train_seconds": round(train_s, 2),
        "compile_seconds": round(compile_s, 2),
        "timings_ms": timings,
        "fhe_execute_median_ms": timings["fhe_execute"]["median_ms"],
        "encryption_tax_x": enc_tax,
    }

    tag = f"{hw['hostname']}_{hw['arch_label']}".replace(" ", "_")
    json_path = os.path.join(RESULTS_DIR, f"pilot_{tag}.json")
    csv_path = os.path.join(RESULTS_DIR, f"pilot_{tag}.csv")

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    with open(csv_path, "w") as f:
        f.write("metric,value\n")
        f.write(f"hostname,{hw['hostname']}\n")
        f.write(f"arch,{hw['arch_label']}\n")
        f.write(f"cpu,{hw['cpu_model']}\n")
        f.write(f"avx512,{hw['has_avx512']}\n")
        f.write(f"plaintext_median_ms,{timings['plaintext']['median_ms']}\n")
        f.write(f"fhe_execute_median_ms,{timings['fhe_execute']['median_ms']}\n")
        f.write(f"encryption_tax_x,{enc_tax}\n")

    print("\n" + "=" * 60)
    print("RESULT SUMMARY")
    print("=" * 60)
    print(f"Plaintext median:     {timings['plaintext']['median_ms']} ms")
    print(f"FHE-execute median:   {timings['fhe_execute']['median_ms']} ms")
    print(f"Encryption tax:       {enc_tax}x slower than plaintext")
    print(f"\nWritten: {json_path}")
    print(f"Written: {csv_path}")
    print("\nRun this on BOTH machines, then run:  python src/compare.py")


if __name__ == "__main__":
    main()
