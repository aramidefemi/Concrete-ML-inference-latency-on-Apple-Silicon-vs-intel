# HE Inference Pilot — Hardware Benchmark

A minimal pilot to measure lightweight homomorphic encryption (HE) inference latency on two different machines, to decide which should act as the **inference server** and which as the **load-testing client** in the main dissertation experiment.

**Project:** Never Decrypt — Benchmarking Lightweight Privacy-Preserving Inference at Web API Scale
**Purpose of this pilot:** determine the faster HE inference host across two CPU architectures (Apple Silicon M5 vs Intel x86 i7).

---

## What this pilot does

1. Trains a small feedforward neural network on the ULB Credit Card Fraud Detection dataset (plaintext).
2. Compiles it to an FHE-equivalent circuit using **Concrete ML** (TFHE-based).
3. Runs single-request inference in three modes and times each:
   - **plaintext** (baseline sklearn/torch inference)
   - **FHE simulation** (correctness check, no real crypto cost)
   - **FHE execute** (real encrypted inference — the number that matters)
4. Writes a JSON + CSV result file tagged with the machine's hardware profile.

Run it on **both machines**, then compare the two `results/*.json` files. The machine with the lower `fhe_execute_ms` becomes your inference server.

---

## Why this matters

HE performance depends heavily on CPU architecture and vector-instruction support (AVX-512 on x86, NEON/AMX on Apple Silicon). Concrete ML and its Intel HEXL acceleration layer are x86-optimised, so the newer ARM machine is **not guaranteed** to be faster. This pilot measures it rather than assuming.

---

## Setup

### 1. Clone and enter
```bash
git clone <your-repo-url>
cd he-pilot
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Apple Silicon note:** Concrete ML wheels are published for macOS ARM, but if `pip install concrete-ml` fails, see `docs/APPLE_SILICON.md` for the fallback (Rosetta or Docker).

### 4. Get the dataset
Download `creditcard.csv` from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place it in the project root as `creditcard.csv`.
(The script also auto-generates a small synthetic fallback if the file is missing, so you can smoke-test the pipeline without the real data.)

---

## Run the pilot

```bash
python src/run_pilot.py
```

This prints a summary and writes:
- `results/pilot_<hostname>_<arch>.json`
- `results/pilot_<hostname>_<arch>.csv`

Run it on **both** machines, commit the results, and compare.

---

## Compare results across machines

Once you have both result files in `results/`:
```bash
python src/compare.py
```
This prints a side-by-side table and tells you which machine to use as the inference server.

---

## Files

| File | Purpose |
|------|---------|
| `src/run_pilot.py` | Main pilot — train, compile, time inference |
| `src/compare.py` | Compare result files across machines |
| `src/hwinfo.py` | Collects hardware/architecture profile |
| `requirements.txt` | Python dependencies |
| `docs/APPLE_SILICON.md` | Fallback install notes for M-series Macs |
| `results/` | Output JSON/CSV per machine |

---

## Interpreting the output

The decisive field is **`fhe_execute_ms`** (median real encrypted inference latency).

- Lower `fhe_execute_ms` → that machine is the **inference server**.
- The other machine becomes the **k6 load-testing client**.

Record **both** machines' numbers in your methodology chapter — the cross-architecture comparison is itself a small reportable finding about HE portability.
