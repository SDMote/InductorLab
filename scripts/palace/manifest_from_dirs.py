#!/usr/bin/env python3
"""
manifest_from_dirs.py
=====================
Rebuild a sweep's manifest.csv from the Palace model dirs that were already built, for when
the generator is killed (e.g. SLURM TIMEOUT) before its single end-of-run manifest write.

Each model dir is named sweep_L{L}_N{N}_{w}_{s}_{d_out}_data, encoding the (grid-snapped)
geometry that was actually drawn. Predictions are recomputed from generate_lhs_sweep.predict(),
so the result is what the generator would have written for the completed subset. Only dirs
with a config.json (a complete build) are included; the half-built last model is skipped.

    python3 scripts/manifest_from_dirs.py lhs_wides
"""
import csv
import re
import sys
from pathlib import Path

import generate_lhs_sweep as gls

REPO = gls.REPO
_NAME = re.compile(r"sweep_L\d+_N(\d+)_([\dp]+)_([\dp]+)_([\dp]+)_data$")


def main():
    sweep = sys.argv[1] if len(sys.argv) > 1 else "lhs_wides"
    OUT = REPO / "scripts" / ".out" / sweep
    pm = OUT / "palace_model"
    if not pm.is_dir():
        print(f"no model dir: {pm}"); return
    rows = []
    for d in sorted(pm.glob("*_data")):
        if not (d / "config.json").exists():        # skip the incomplete (killed mid-build) model
            continue
        m = _NAME.search(d.name)
        if not m:
            continue
        n = int(m[1])
        w = float(m[2].replace("p", ".")); s = float(m[3].replace("p", "."))
        do = float(m[4].replace("p", "."))
        p = gls.predict(n, w * 1e-6, s * 1e-6, do * 1e-6)
        model_dir = str(d.resolve().relative_to(REPO))
        rows.append(dict(idx=len(rows) + 1, target_nH=round(p["L"] * 1e9), N=n,
                         w_um=round(w, 4), s_um=round(s, 4), d_out_um=round(do, 3),
                         d_avg_um=round(p["d_avg"] * 1e6, 3),
                         pred_L_nH=round(p["L"] * 1e9, 4), pred_Cs_fF=round(p["C_s"] * 1e15, 3),
                         pred_Cp_fF=round(p["C_p"] * 1e15, 3), pred_Ctot_fF=round(p["C_tot"] * 1e15, 3),
                         pred_Q=round(p["Q"], 3), pred_SRF_GHz=round(p["fsr"] / 1e9, 4),
                         model_dir=model_dir))
    if not rows:
        print("No completed model dirs found."); return
    with open(OUT / "manifest.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        wtr.writeheader(); wtr.writerows(rows)
    print(f"Rebuilt manifest from {len(rows)} built models: {OUT/'manifest.csv'}")


if __name__ == "__main__":
    main()
