#!/usr/bin/env python3
"""
generate_sweep.py
=================
Fixed-frequency design sweep (Hershenson Fig. 5 style): for each target L = 1..20 nH, design a
symmetric octagonal inductor with the OPERATING-L-targeted GP and the Palace-calibrated
coefficients (spiral_cturn.design_davg_operating + ASITIC/coefficients_sym_sg13g2_palace.py),
build its Palace model, and -- via the array job -- simulate it to compare the GP's predicted
L/Q/SRF against the full-wave EM truth across the whole design range.

Each design targets L_operating == target at FREQ (so a "10 nH" coil is 10 nH at the operating
frequency, not at DC). Models are built robustly (subprocess per-model timeout + incremental
manifest, reused from generate_lhs_sweep), so a wide-spacing high-L coil that hangs gmsh is
skipped, not fatal. NOT simulated here -- run on CLEPS:
    sbatch cleps/generate_sweep.sh
    sbatch --dependency=afterany:<gid> cleps/run_lhs_array.sh sweep

    python3 scripts/generate_sweep.py                 # L = 1..20 nH @ 2.5 GHz
    python3 scripts/generate_sweep.py --freq 3.0 --lmax 12
"""
import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import generate_sweep_palace as gsp
import generate_lhs_sweep as gls      # build_with_timeout, _write_manifest, PER_MODEL_TIMEOUT
import spiral_cturn as sct

REPO = gsp.REPO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmin", type=int, default=1)
    ap.add_argument("--lmax", type=int, default=20)
    ap.add_argument("--freq", type=float, default=2.5, help="operating frequency [GHz]")
    ap.add_argument("--srf", type=float, default=7.0, help="SRF floor [GHz]")
    a = ap.parse_args()

    OUT = REPO / "scripts" / ".out" / "sweep"
    OUT.mkdir(parents=True, exist_ok=True)
    sct.FREQ = a.freq * 1e9                              # design + tank boost at this freq
    gsp.FSWEEP = True
    gsp.FSWEEP_STOP = 15.0e9
    gsp.FREQ = a.freq * 1e9                              # Palace anchor
    print(f"sweep L={a.lmin}..{a.lmax} nH @ {a.freq} GHz (operating-targeted, Palace coeffs)  ->  {OUT.relative_to(REPO)}\n")

    rows = []
    skipped = []
    for L in range(a.lmin, a.lmax + 1):
        d = sct.best_int(L, _design=sct.design_davg_operating, min_srf_hz=a.srf * 1e9)
        if d is None:
            skipped.append(f"{L}(infeasible)"); print(f"  L={L:2d}  INFEASIBLE (SRF>={a.srf} GHz)"); continue
        N = round(d["n"])
        sol = SimpleNamespace(n=N, w=d["w"], s=d["s"], d_out=d["d_out"], L=d["L_op"])
        ind_name, info = gls.build_with_timeout(sol, OUT, gls.PER_MODEL_TIMEOUT)
        if ind_name is None:
            skipped.append(f"{L}({info})"); print(f"  L={L:2d}  SKIP ({info})"); continue
        model_dir = str(Path(info).resolve().relative_to(REPO))
        rows.append(dict(idx=len(rows) + 1, target_nH=L, op_GHz=a.freq, N=N,
                         w_um=round(d["w"] * 1e6, 4), s_um=round(d["s"] * 1e6, 4),
                         d_out_um=round(d["d_out"] * 1e6, 3), gp_LDC_nH=round(d["L"] * 1e9, 4),
                         gp_Lop_nH=round(d["L_op"] * 1e9, 4), gp_Q=round(d["Q"], 3),
                         gp_SRF_GHz=round(d["f_sr"] / 1e9, 3), model_dir=model_dir))
        gls._write_manifest(OUT, rows)
        print(f"  [{len(rows):2d}] L={L:2d}nH  N={N} w={d['w']*1e6:5.1f} s={d['s']*1e6:5.1f} "
              f"d_out={d['d_out']*1e6:5.0f}um  Q={d['Q']:4.1f} SRF={d['f_sr']/1e9:.2f}  -> {ind_name}")

    if not rows:
        print("No models built."); return
    print(f"\nWrote {len(rows)} models + manifest: {OUT/'manifest.csv'}"
          + (f"  (skipped: {skipped})" if skipped else ""))


if __name__ == "__main__":
    main()
