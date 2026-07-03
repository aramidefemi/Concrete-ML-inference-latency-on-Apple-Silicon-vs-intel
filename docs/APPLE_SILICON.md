# Apple Silicon (M-series) Install Notes

Concrete ML is optimised for x86 and its acceleration layer (Intel HEXL) targets
AVX-512. On Apple Silicon (M1–M5, `arm64`), the standard `pip install concrete-ml`
usually works with recent releases, but if it fails, use one of the fallbacks below.

---

## First: just try the normal install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
If `python src/run_pilot.py` runs, you're done. Only use the fallbacks if the
install or run fails.

---

## Fallback A — Rosetta 2 (run x86 Python under emulation)

This runs an x86 Python toolchain on Apple Silicon via Rosetta. Slower than
native, but maximises Concrete ML / HEXL compatibility.

```bash
# Install Rosetta if not present
softwareupdate --install-rosetta --agree-to-license

# Launch an x86 shell
arch -x86_64 zsh

# Inside that shell, use an x86 Homebrew Python
arch -x86_64 /usr/local/bin/python3 -m venv .venv-x86
source .venv-x86/bin/activate
pip install -r requirements.txt
python src/run_pilot.py
```

> If you use Fallback A, **note it in your results** — Rosetta emulation affects
> latency and is itself a confound worth documenting in the methodology.

---

## Fallback B — Docker (clean x86 Linux environment)

Most reproducible option. Runs an x86_64 Linux container.

```bash
# From the project root
docker run --platform linux/amd64 -it \
  -v "$PWD":/app -w /app \
  python:3.11-slim bash

# Inside the container:
pip install -r requirements.txt
python src/run_pilot.py
```

The result file will be tagged with the container's hostname/arch. Rename it so
you can tell it apart from the native run, e.g.:
```bash
mv results/pilot_*.json results/pilot_m5_docker_amd64.json
```

---

## Which fallback should I use?

- **Goal is a fair cross-architecture comparison** → try native ARM first
  (Fallback none). A native ARM number vs a native x86 number is the cleanest
  comparison for your methodology chapter.
- **Native ARM install fails entirely** → Fallback B (Docker), and clearly label
  the result as emulated x86 rather than native ARM.
- **Avoid mixing** native-ARM on one machine and Docker-x86 on the other without
  labelling it — the comparison only means something if you know exactly what
  ran where.

---

## Recording what you ran

Whatever path you take, the `hardware` block in the result JSON captures arch,
CPU model, and AVX-512 support automatically. Add a one-line note to your lab
log stating: native vs Rosetta vs Docker, so the methodology write-up is accurate.
