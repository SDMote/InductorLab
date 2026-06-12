#!/usr/bin/env python3
"""
asitic_sym_sweep.py
===================
Symmetric octagonal (center-tapped) SG13G2 inductor sweep with the CORRECTED
sympoly flow, producing process-specific monomial coefficients.

Corrections vs the original sympoly sweep:
  * radius = 1.05*(d_out/2): sympoly's radius undersizes the coil ~5% if you
    feed the apothem d_out/2 directly (verified vs KlayoutDrawing GDS).
  * fit on the BUILT geometry: extract d_out/d_avg from the CIF turn positions
    (de-rotated 22.5deg), since sympoly's geom total length is contaminated by
    the center-tap crossovers.

Geometry: octagon on TopMetal2, crossovers on TopMetal1, transition gap
  e = w + s + (w+2s)/(1+sqrt(2))  (= sympoly ILEN), integer turns only.

Run from the ASITIC/ directory:
  python3 asitic_sym_sweep.py --n-samples 1500 --workers 8
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
                    format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Reference coefficient sets for context (not a validation target — symmetric
# coils have no Mohan reference).
MOHAN_OCT = {"beta": 1.33e-3, "a1": -1.21, "a2": -0.163, "a3": 2.43, "a4": 1.75, "a5": -0.049}

# Parameter space (SG13G2 TopMetal2 DRC: w>=2, s>=2; realistic d_out).
DOUT_MIN, DOUT_MAX = 100.0, 480.0
W_MIN, W_MAX_FRAC = 2.0, 0.3
S_MIN, S_MAX_FRAC = 2.0, 3.0
N_CHOICES = [1, 2, 3, 4, 5, 6]          # symmetric coils -> integer turns
L_MIN, L_MAX = 0.3, 100.0               # nH
GRID = 0.005
RADIUS_FACTOR = 0.525                   # 1.05*(d_out/2)/d_out, sympoly radius
PER_INDUCTOR_TIMEOUT = 12
_TH = math.radians(22.5)
_CT, _ST = math.cos(_TH), math.sin(_TH)


def _snap(v):
    return round(v / GRID) * GRID


@dataclass
class P:
    d_out: float
    w: float
    s: float
    n: float

    @property
    def e(self):
        return self.w + self.s + (self.w + 2.0 * self.s) / (1.0 + math.sqrt(2.0))

    @property
    def d_in_nom(self):
        return self.d_out - 2.0 * (self.n * self.w + (self.n - 1.0) * self.s)

    def is_valid(self):
        if self.w < W_MIN or self.s < S_MIN:
            return False
        if self.w > W_MAX_FRAC * self.d_out or self.s > S_MAX_FRAC * self.w:
            return False
        # central transition gap e must fit in the inner opening, with margin
        if self.d_in_nom <= self.e + 2.0 * self.w:
            return False
        return True


def generate_samples(n_samples, seed):
    log.info("Generating LHS samples (seed=%d, target=%d, symmetric octagon)...", seed, n_samples)
    params, total, batch = [], 0, 0
    pool = n_samples * 6
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
            n = N_CHOICES[min(int(r[3] * len(N_CHOICES)), len(N_CHOICES) - 1)]
            p = P(_snap(d_out), _snap(w), _snap(s), n)
            if p.is_valid():
                params.append(p)
                if len(params) >= n_samples:
                    break
    params = params[:n_samples]
    log.info("Generated %d valid samples (from %d draws, %d batch(es))", len(params), total, batch)
    return params


_RE_DC = re.compile(r"DC Inductance L\(m\)\s*=\s*([\d.eE+\-]+)\s*(nH|pH|µH|uH)", re.IGNORECASE)
# AC: pix (symmetric pi-model) reports "L = X nH R = Y"; negative R => bad extraction.
_RE_AC = re.compile(r"Pi Model at f=.*?\bL\s*=\s*([\d.eE+\-]+)\s*(nH|pH|µH|uH)\s*R\s*=\s*([\d.eE+\-]+)",
                    re.DOTALL | re.IGNORECASE)
_RE_FSR = re.compile(r"Est\.?\s*Resonance\s*=\s*([\d.]+)\s*GHz", re.IGNORECASE)


def _cif_geom(cif_text, w):
    """Built (d_out, d_avg) from CIF turn positions on the de-rotated horizontal
    axis. Returns None if the octagon turns can't be cleanly recovered."""
    polys = []
    for m in re.finditer(r"P([0-9 \t-]+);", cif_text):
        v = [int(x) for x in re.findall(r"-?\d+", m.group(1))]
        polys.append([(((v[i] * 0.01 - 256) * _CT + (v[i + 1] * 0.01 - 256) * _ST),
                       (-(v[i] * 0.01 - 256) * _ST + (v[i + 1] * 0.01 - 256) * _CT))
                      for i in range(0, len(v), 2)])
    ivs = []
    for poly in polys:
        xs = []
        k = len(poly)
        for i in range(k):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % k]
            if (y1 <= 0 < y2) or (y2 <= 0 < y1):
                xs.append(x1 + (x2 - x1) * (0 - y1) / (y2 - y1))
        xs.sort()
        for j in range(0, len(xs) - 1, 2):
            ivs.append((xs[j], xs[j + 1]))
    right = sorted([(a, b) for a, b in ivs if a > 1])
    merged = []
    for a, b in right:
        if merged and a <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    # keep only intervals shaped like a turn (width ~ w)
    turns = [(a, b) for a, b in merged if 0.7 * w < (b - a) < 1.4 * w]
    if not turns:
        return None
    a_out = max(b for a, b in turns)
    a_in = min(a for a, b in turns)
    d_out = 2 * a_out
    d_avg = a_out + a_in
    return d_out, d_avg


def _build_script(p, cif_path, freq):
    radius = RADIUS_FACTOR * p.d_out
    # AC: measure DC too, as a sanity reference for the pi-model extraction.
    calc = f"inductance m\npix m {freq}" if freq else "inductance m"
    return ("sympoly\nm\n"
            f"{radius:.3f}\n{p.w:.3f}\nTopMetal2\nTopMetal1\n{p.s:.3f}\n{p.e:.3f}\n8\n{int(p.n)}\nc\n"
            f"cifsave m {cif_path}\n{calc}\ndelete m\nexit\n")


def run_one(asitic_bin, tek_file, p, freq=None):
    cmd_file = out_file = cif_file = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cmd", delete=False) as f:
            cmd_file = f.name
        cif_file = cmd_file + ".cif"
        out_file = cmd_file + ".out"
        with open(cmd_file, "w") as f:
            f.write(_build_script(p, cif_file, freq))
        try:
            with open(cmd_file) as fin, open(out_file, "w") as fout:
                subprocess.run([asitic_bin, "-g", "-t", tek_file], stdin=fin, stdout=fout,
                               stderr=subprocess.STDOUT, start_new_session=True,
                               timeout=PER_INDUCTOR_TIMEOUT)
        except subprocess.TimeoutExpired:
            return None
        text = open(out_file).read()

        def _nH(mo):
            val = float(mo.group(1))
            u = mo.group(2).lower()
            return val / 1000.0 if u == "ph" else (val * 1000.0 if u in ("uh", "µh") else val)

        if freq:
            m = _RE_AC.search(text)
            if not m or float(m.group(3)) <= 0:   # require positive R (good extraction)
                return None
            fsr = _RE_FSR.search(text)             # drop coils too near self-resonance
            if fsr and float(fsr.group(1)) < 2.0 * float(freq):
                return None
            l = _nH(m)
            mdc = _RE_DC.search(text)              # AC must be a sane multiple of DC,
            if not mdc:                            # else the pi-extraction is broken
                return None
            ldc = _nH(mdc)
            if not (0.8 * ldc < l < 2.5 * ldc):
                return None
        else:
            m = _RE_DC.search(text)
            if not m:
                return None
            l = _nH(m)
        if not (L_MIN < l < L_MAX):
            return None
        if not os.path.exists(cif_file):
            return None
        g = _cif_geom(open(cif_file).read(), p.w)
        if g is None:
            return None
        d_out_b, d_avg_b = g
        return (p, l, d_out_b, d_avg_b)
    except Exception as e:
        log.debug("run_one error: %s", e)
        return None
    finally:
        for fp in (cmd_file, out_file, cif_file):
            if fp and os.path.exists(fp):
                try:
                    os.unlink(fp)
                except OSError:
                    pass


def _worker(args):
    return run_one(*args)


def sweep(params, asitic_bin, tek_file, workers, out_path, freq=None):
    log.info("Sweeping %d symmetric inductors (workers=%d, freq=%s)...",
             len(params), workers, freq or "DC")
    args = [(asitic_bin, tek_file, p, freq) for p in params]
    results, t0, done = [], time.time(), 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_worker, a) for a in args]
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                results.append(r)
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


FIELDS = ["d_out", "w", "s", "n", "d_out_built", "d_avg_built", "L_nH"]


def save(results, path):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        for p, l, do, da in results:
            wr.writerow({"d_out": p.d_out, "w": p.w, "s": p.s, "n": int(p.n),
                         "d_out_built": round(do, 3), "d_avg_built": round(da, 3),
                         "L_nH": round(l, 6)})


def load(path):
    out = []
    for row in csv.DictReader(open(path, newline="")):
        p = P(float(row["d_out"]), float(row["w"]), float(row["s"]), float(row["n"]))
        out.append((p, float(row["L_nH"]), float(row["d_out_built"]), float(row["d_avg_built"])))
    return out


def fit(results):
    X, y = [], []
    for p, l, do, da in results:
        if min(l, do, da, p.w, p.s, p.n) <= 0:
            continue
        X.append([1, math.log(do), math.log(p.w), math.log(da), math.log(p.n), math.log(p.s)])
        y.append(math.log(l))
    X, y = np.array(X), np.array(y)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    beta = math.exp(c[0])
    a1, a2, a3, a4, a5 = c[1:]
    err = 100 * np.abs(np.exp(X @ c) - np.exp(y)) / np.exp(y)
    fitd = {"beta": beta, "a1": a1, "a2": a2, "a3": a3, "a4": a4, "a5": a5,
            "n_points": len(y), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "median": float(np.median(err))}
    log.info("")
    log.info("=" * 60)
    log.info("SG13G2 SYMMETRIC OCTAGON  monomial fit  (built geometry)")
    log.info("=" * 60)
    log.info("  %-10s %14s %14s", "coeff", "ours", "Mohan oct")
    for name, key in [("beta", "beta"), ("a1 d_out", "a1"), ("a2 w", "a2"),
                      ("a3 d_avg", "a3"), ("a4 n", "a4"), ("a5 s", "a5")]:
        log.info("  %-10s %14.4e %14.4e", name, fitd[key.split()[0] if " " in key else key],
                 MOHAN_OCT[key.split()[0] if " " in key else key])
    log.info("")
    log.info("  N points : %d   RMSE : %.2f%%   median : %.2f%%",
             fitd["n_points"], fitd["rmse"], fitd["median"])
    log.info("=" * 60)
    return fitd


def save_coeffs(fitd, path):
    with open(path, "w") as f:
        f.write("# SG13G2 SYMMETRIC octagonal inductance coefficients (sympoly, built-geometry fit)\n")
        f.write("# L_nH = BETA * d_out**A1 * w**A2 * d_avg**A3 * n**A4 * s**A5  (microns, nH)\n")
        f.write(f"# Fit: RMSE={fitd['rmse']:.2f}%  N={fitd['n_points']}\n\n")
        f.write(f"BETA = {fitd['beta']:.6e}\n")
        for i, k in enumerate(["a1", "a2", "a3", "a4", "a5"], 1):
            f.write(f"A{i}   = {fitd[k]:.6f}\n")
        f.write("\n\ndef inductance_nH(d_out, w, s, n):\n")
        f.write("    d_avg = d_out - n * (w + s)\n")
        f.write("    return BETA * d_out**A1 * w**A2 * d_avg**A3 * n**A4 * s**A5\n")
    log.info("Coefficients saved to %s", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asitic", default="./asitic_linux")
    ap.add_argument("--tek", default="./ihp_sg13g2.tek")
    ap.add_argument("--n-samples", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    ap.add_argument("--load", default=None)
    ap.add_argument("--freq", default=None,
                    help="GHz for AC inductance via pix (symmetric pi-model); omit for DC")
    a = ap.parse_args()
    tag = f"_{a.freq}ghz" if a.freq else ""
    if a.output is None:
        a.output = f"results_sym_sg13g2{tag}.csv"
    coeff_path = f"coefficients_sym_sg13g2{tag}.py"
    if a.load:
        results = load(a.load)
        log.info("Loaded %d results from %s", len(results), a.load)
    else:
        for path, label in [(a.asitic, "ASITIC binary"), (a.tek, ".tek file")]:
            if not Path(path).exists():
                log.error("%s not found: %s", label, path)
                sys.exit(1)
        params = generate_samples(a.n_samples, a.seed)
        if a.freq:
            # pix reads/writes a Green's-function cache <tek>_<freq>.dat. Precompute
            # it once (serially, long timeout) so parallel workers reuse it w/o racing.
            cache = f"{Path(a.tek).name}_{float(a.freq):.2f}.dat"
            if not Path(cache).exists():
                log.info("Precomputing %s GHz Green's function (one-time, slow)...", a.freq)
                pre = (f"sympoly\np\n78.75\n8\nTopMetal2\nTopMetal1\n4\n20\n8\n3\nc\n"
                       f"pix p {a.freq}\ndelete p\nexit\n")
                with open("/tmp/_pre.cmd", "w") as f:
                    f.write(pre)
                with open("/tmp/_pre.cmd") as fin, open("/tmp/_pre.out", "w") as fout:
                    subprocess.run([a.asitic, "-g", "-t", a.tek], stdin=fin, stdout=fout,
                                   stderr=subprocess.STDOUT, start_new_session=True, timeout=400)
                log.info("Green's function cache ready: %s", Path(cache).exists())
        results = sweep(params, a.asitic, a.tek, a.workers, a.output, a.freq)
        if len(results) < 50:
            log.error("Too few results (%d).", len(results))
            sys.exit(1)
        save(results, a.output)
        log.info("Saved %d results to %s", len(results), a.output)
    fitd = fit(results)
    save_coeffs(fitd, coeff_path)


if __name__ == "__main__":
    main()
