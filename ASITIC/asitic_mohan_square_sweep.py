#!/usr/bin/env python3
"""
asitic_mohan_square_sweep.py
============================
Replicate Mohan 1999 (Table III/IV) SQUARE-inductor sweep using the CORRECT
ASITIC command sequence — the dedicated `square` command (outer dimension
edge-to-edge) — to check that we reproduce Mohan's published square monomial
coefficients. This validates the methodology before trusting the octagonal flow.

Validated single point: square d_out=279 w=18.3 s=1.9 n=2.75 + exit segment
=> 3.109 nH  (Mohan Table IV #4: 3.10 nH).

Run from the ASITIC/ directory (CMOS.tek loads relative to its TechPath=.):
    python3 asitic_mohan_square_sweep.py --tek CMOS.tek --metal m5 --n-samples 1900
    python3 asitic_mohan_square_sweep.py --exit-metal m4   # include underpass lead
"""

import argparse
import csv
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import qmc

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Mohan 1999 Table III — SQUARE monomial coefficients (the validation target).
#   L_nH = beta * d_out^a1 * w^a2 * d_avg^a3 * n^a4 * s^a5   (microns, nH)
MOHAN_SQUARE = {"beta": 1.62e-3, "a1": -1.21, "a2": -0.147,
                "a3": 2.40, "a4": 1.78, "a5": -0.030}

# Mohan parameter ranges (Section I): d_out 100-480, w 2..0.3*d_out, s 2..3*w.
DOUT_MIN, DOUT_MAX = 100.0, 480.0
W_MIN, W_MAX_FRAC = 2.0, 0.3
S_MIN, S_MAX_FRAC = 2.0, 3.0
N_MIN, N_MAX = 1.0, 12.0           # turns (continuous; d_in & L bounds prune it)
DIN_MIN = 10.0                     # need an inner hole
L_MIN, L_MAX = 0.5, 100.0          # nH, Mohan kept inductors in this band
SIDES = 4
PER_INDUCTOR_TIMEOUT = 10
GRID = 0.005


def _snap(v):
    return round(v / GRID) * GRID


@dataclass
class P:
    d_out: float
    w: float
    s: float
    n: float

    @property
    def d_in(self):
        # n traces with (n-1) spacings between them (Mohan convention).
        return self.d_out - 2.0 * (self.n * self.w + (self.n - 1.0) * self.s)

    @property
    def d_avg(self):
        return 0.5 * (self.d_out + self.d_in)

    def is_valid(self):
        if self.w < W_MIN or self.s < S_MIN:
            return False
        if self.w > W_MAX_FRAC * self.d_out:
            return False
        if self.s > S_MAX_FRAC * self.w:
            return False
        if self.d_in < DIN_MIN:
            return False
        return True


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def generate_samples(n_samples, seed):
    log.info("Generating LHS samples (seed=%d, target=%d, Mohan square ranges)...",
             seed, n_samples)
    params, total, batch = [], 0, 0
    pool = n_samples * 4
    while len(params) < n_samples:
        raw = qmc.LatinHypercube(d=4, seed=seed + batch).random(pool)
        total += pool
        batch += 1
        for r in raw:
            d_out = DOUT_MIN + r[0] * (DOUT_MAX - DOUT_MIN)
            w_lo = W_MIN / d_out
            if w_lo >= W_MAX_FRAC:
                continue
            w = d_out * (w_lo + r[1] * (W_MAX_FRAC - w_lo))
            s_lo = S_MIN / w
            if s_lo >= S_MAX_FRAC:
                continue
            s = w * (s_lo + r[2] * (S_MAX_FRAC - s_lo))
            n = N_MIN + r[3] * (N_MAX - N_MIN)
            p = P(_snap(d_out), _snap(w), _snap(s), round(n, 3))
            if p.is_valid():
                params.append(p)
                if len(params) >= n_samples:
                    break
    params = params[:n_samples]
    log.info("Generated %d valid samples (from %d draws, %d batch(es))",
             len(params), total, batch)
    return params


# ---------------------------------------------------------------------------
# ASITIC interface  (`square` command + optional exit segment)
# ---------------------------------------------------------------------------

_RE_DC = re.compile(r"DC Inductance L\(i(\d+)\)\s*=\s*([\d.eE+\-]+)\s*(nH|pH|µH|uH)",
                    re.IGNORECASE)


def _build_script(batch, metal, exit_metal):
    lines = []
    for idx, p in batch:
        name = f"i{idx}"
        # square prompts: Name, Outer-dim(edge-to-edge), Width, Spacing, Turns,
        # Metal, "Add exit segment?", [Exit metal], Origin.
        lines += ["square", name, f"{p.d_out:.3f}", f"{p.w:.3f}",
                  f"{p.s:.3f}", f"{p.n:.4f}", metal]
        if exit_metal:
            lines += ["y", exit_metal]
        else:
            lines += ["n"]
        lines += ["c", f"inductance {name}", f"delete {name}"]
    lines.append("exit")
    return "\n".join(lines) + "\n"


def _parse(output, batch):
    by_idx = {idx: p for idx, p in batch}
    out, seen = [], set()
    for sidx, val, unit in _RE_DC.findall(output):
        idx = int(sidx)
        if idx not in by_idx or idx in seen:
            continue
        try:
            l = float(val)
        except ValueError:
            continue
        u = unit.lower()
        if u == "ph":
            l /= 1000.0
        elif u in ("uh", "µh"):
            l *= 1000.0
        if L_MIN < l < L_MAX:
            out.append((idx, by_idx[idx], l))
            seen.add(idx)
    return out


def run_batch(asitic_bin, tek_file, batch, metal, exit_metal):
    script = _build_script(batch, metal, exit_metal)
    timeout_s = PER_INDUCTOR_TIMEOUT * len(batch)
    cmd_file = out_file = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cmd", delete=False) as f:
            cmd_file = f.name
            f.write(script)
        out_file = cmd_file + ".out"
        try:
            with open(cmd_file) as fin, open(out_file, "w") as fout:
                subprocess.run([asitic_bin, "-g", "-t", tek_file], stdin=fin,
                               stdout=fout, stderr=subprocess.STDOUT,
                               start_new_session=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            pass  # recover whatever finished before the hang
        with open(out_file) as f:
            return _parse(f.read(), batch)
    except Exception as e:
        log.error("Batch error: %s", e)
        return []
    finally:
        for fp in (cmd_file, out_file):
            if fp and os.path.exists(fp):
                try:
                    os.unlink(fp)
                except OSError:
                    pass


def _worker(args):
    return run_batch(*args)


def sweep(params, asitic_bin, tek_file, metal, exit_metal, workers, out_path):
    indexed = list(enumerate(params))
    log.info("Sweeping %d square inductors (workers=%d, exit_segment=%s)...",
             len(params), workers, bool(exit_metal))
    args = [(asitic_bin, tek_file, [b], metal, exit_metal) for b in indexed]
    results, t0, done = [], time.time(), 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_worker, a) for a in args]
        for fut in as_completed(futs):
            results.extend(fut.result())
            done += 1
            if done % 100 == 0 or done == len(args):
                el = time.time() - t0
                eta = (len(args) - done) / (done / el) if done else 0
                log.info("[%4d/%4d] results=%4d  %.0f%%  ETA %.0fs",
                         done, len(args), len(results), 100 * done / len(args), eta)
                if out_path:
                    save(results, out_path)
    log.info("Sweep complete: %d/%d succeeded (%.1f%%)",
             len(results), len(params), 100 * len(results) / max(len(params), 1))
    return results


# ---------------------------------------------------------------------------
# I/O + fit
# ---------------------------------------------------------------------------

FIELDS = ["idx", "d_out", "w", "s", "n", "d_in", "d_avg", "L_nH"]


def save(results, path):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        for idx, p, l in results:
            wr.writerow({"idx": idx, "d_out": p.d_out, "w": p.w, "s": p.s, "n": p.n,
                         "d_in": round(p.d_in, 3), "d_avg": round(p.d_avg, 3),
                         "L_nH": round(l, 6)})


def load(path):
    out = []
    for row in csv.DictReader(open(path, newline="")):
        p = P(float(row["d_out"]), float(row["w"]), float(row["s"]), float(row["n"]))
        out.append((int(row["idx"]), p, float(row["L_nH"])))
    return out


def fit_and_compare(results):
    rows, y = [], []
    for _, p, l in results:
        if min(l, p.d_avg, p.d_out, p.w, p.s, p.n) <= 0:
            continue
        rows.append([1.0, math.log(p.d_out), math.log(p.w),
                     math.log(p.d_avg), math.log(p.n), math.log(p.s)])
        y.append(math.log(l))
    X, y = np.array(rows), np.array(y)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    beta = math.exp(c[0])
    a1, a2, a3, a4, a5 = c[1:]
    err = 100 * np.abs(np.exp(X @ c) - np.exp(y)) / np.exp(y)

    log.info("")
    log.info("=" * 64)
    log.info("SQUARE MONOMIAL FIT  vs  Mohan 1999 Table III")
    log.info("=" * 64)
    log.info("  %-10s %14s %14s %9s", "Coeff", "ours", "Mohan", "delta%")
    log.info("  " + "-" * 50)
    pairs = [("beta", beta, MOHAN_SQUARE["beta"]), ("a1 d_out", a1, MOHAN_SQUARE["a1"]),
             ("a2 w", a2, MOHAN_SQUARE["a2"]), ("a3 d_avg", a3, MOHAN_SQUARE["a3"]),
             ("a4 n", a4, MOHAN_SQUARE["a4"]), ("a5 s", a5, MOHAN_SQUARE["a5"])]
    for name, ours, mo in pairs:
        d = 100 * (ours - mo) / abs(mo) if mo else float("nan")
        log.info("  %-10s %14.4e %14.4e %+8.1f%%", name, ours, mo, d)
    log.info("")
    log.info("  N points        : %d", len(y))
    log.info("  fit RMSE        : %.2f%%", math.sqrt(np.mean(err ** 2)))
    log.info("  fit median err  : %.2f%%", float(np.median(err)))
    log.info("=" * 64)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asitic", default="./asitic_linux")
    ap.add_argument("--tek", default="CMOS.tek")
    ap.add_argument("--metal", default="m5")
    ap.add_argument("--exit-metal", default=None,
                    help="underpass metal for exit segment (e.g. m4); omit for bare coil")
    ap.add_argument("--n-samples", type=int, default=1900)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="results_square_cmos.csv")
    ap.add_argument("--load", default=None)
    return ap.parse_args()


def main():
    a = parse_args()
    if a.load:
        results = load(a.load)
        log.info("Loaded %d results from %s", len(results), a.load)
    else:
        for path, label in [(a.asitic, "ASITIC binary"), (a.tek, ".tek file")]:
            if not Path(path).exists():
                log.error("%s not found: %s", label, path)
                sys.exit(1)
        params = generate_samples(a.n_samples, a.seed)
        results = sweep(params, a.asitic, a.tek, a.metal, a.exit_metal,
                        a.workers, a.output)
        if len(results) < 50:
            log.error("Too few results (%d) to fit.", len(results))
            sys.exit(1)
        save(results, a.output)
        log.info("Saved %d results to %s", len(results), a.output)
    fit_and_compare(results)


if __name__ == "__main__":
    main()
