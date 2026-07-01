#!/usr/bin/env python3
"""
generate_lhs_sweep.py
=====================
A Latin-hypercube sweep of DIRECTLY-SAMPLED octagonal-inductor geometries (not GP
solutions) to break the n / d_avg collinearity of the L-optimal designs.

The cturn sweeps walked the GP's optimal locus, where corr(log n, log d_avg) = -0.83, so
the inter-turn cap's n vs d_avg exponents are unidentifiable (a free monomial overfits to
n^4.3). Here we sample the four geometric DOF *independently* and reject only the
infeasible:

    n      in {2..8}          (turns; >=2 so there IS an inter-turn cap)
    w      in [2, 28] um       (TopMetal2 w_min = 2)
    s      in [2, 7] um        (TopMetal2 s_min = 2)
    d_avg  in [60, 600] um     (average diameter, sampled ABSOLUTE -> independent of n)

    d_out = d_avg + (n*w + (n-1)*s)        d_in = d_avg - (n*w + (n-1)*s)
    drawable:  d_in >= e + 2w,  e = w+s+(w+2s)/(1+sqrt2)
    area cap:  d_out <= 800 um   ("around (800 um)^2" -- only blocks the truly huge coils)

For each accepted geometry we forward-evaluate our models (inductance monomial, C_s =
overlap + inter-turn at gamma=0.68, C_p, one-port SRF, Hershenson Q) and store them in the
manifest next to the model dir. Palace runs the adaptive 0.1-15 GHz sweep (band widened
from 10 GHz so more coils' SERIES-branch resonance -> C_BR lands in band).

NOT simulated here -- run on CLEPS with cleps/run_lhs_array.sh.

    python3 scripts/generate_lhs_sweep.py            # ~120 feasible models
    python3 scripts/generate_lhs_sweep.py --n 80 --seed 1
    python3 scripts/generate_lhs_sweep.py --dry-run  # sampling stats only, no models
"""
import argparse
import csv
import math
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import qmc

import generate_sweep_palace as gsp     # build_one, FSWEEP, FREQ, COEFFS, PDK
import spiral_cturn as sct              # GAMMA, _k6, _k7, _K_E

REPO = gsp.REPO
PDK, COEFFS, GAMMA = gsp.PDK, gsp.COEFFS, sct.GAMMA
_MU_0, _EPS_0 = 4 * math.pi * 1e-7, 8.854187e-12
_K_E = sct._K_E
RT2 = math.sqrt(2)

# sampling ranges (um) -- absolute, so n and d_avg are independent
N_MIN, N_MAX = 2, 8
W_MIN, W_MAX = 2.0, 28.0
S_MIN, S_MAX = 2.0, 7.0
DAVG_MIN, DAVG_MAX = 60.0, 600.0
DOUT_CAP = 800.0
BAND_STOP = 15.0e9          # widen 10 -> 15 GHz to catch more series-branch (C_BR) resonances
PER_MODEL_TIMEOUT = 480     # s: kill a model whose gmsh meshing hangs (legit builds ~1-4 min)


# ---- forward model: replicate spiral_cturn.design()'s closed form (no GP solve) ----
def _kconsts():
    omega = 2 * math.pi * gsp.FREQ
    metal, via = PDK.top_metals[0], PDK.top_vias[0]
    sigma_m, t_m = metal.sigma, metal.thickness
    sigma_v, t_via, a_via, b_via = via.sigma, via.thickness, via.width, via.space
    e_ox, t_ox = PDK.eps_r_ox * _EPS_0, PDK.t_ox
    e_sub, t_sub, sigma_sub = PDK.eps_r_sub * _EPS_0, PDK.t_sub, PDK.sigma_sub
    skin_m = math.sqrt(2 / (omega * _MU_0 * sigma_m))
    k1 = (1.0 / (sigma_m * skin_m * (1 - math.exp(-t_m / skin_m))) if skin_m < t_m
          else 1.0 / (sigma_m * t_m))
    k2 = e_ox / (2 * t_ox); k3 = e_ox / t_via
    k4 = e_sub / (2 * t_sub); k5 = 2 * t_sub / sigma_sub
    k6 = sct._k6(k2, k4, k5, omega); k7 = sct._k7(k2, k4, k5, omega)
    skin_v = math.sqrt(2 / (omega * _MU_0 * sigma_v))
    k8 = (2 * t_via / (sigma_v * a_via * skin_v * (1 - math.exp(-a_via / skin_v)) * (a_via + b_via) ** 2)
          if skin_v < a_via else 2 * t_via * (a_via + b_via) ** 2 / (sigma_v * a_via ** 2))
    c_turn = GAMMA * 8 * (RT2 - 1) * _EPS_0 * PDK.eps_r_ox * t_m
    return dict(omega=omega, k1=k1, k3=k3, k6=k6, k7=k7, k8=k8, c_turn=c_turn)


_K = _kconsts()


def predict(n, w_m, s_m, dout_m):
    """Closed-form L, C_s(=C_BR), C_p, C_tot, Q, SRF for a geometry (SI units in)."""
    K = _K; omega = K["omega"]
    davg = dout_m - n * w_m - (n - 1) * s_m
    l = 8 * davg * n / (1 + RT2)
    L = (COEFFS.beta * (dout_m / 1e-6) ** COEFFS.a1 * (w_m / 1e-6) ** COEFFS.a2
         * (davg / 1e-6) ** COEFFS.a3 * n ** COEFFS.a4 * (s_m / 1e-6) ** COEFFS.a5) * 1e-9
    R_s = K["k1"] * l / w_m + K["k8"] * n / w_m ** 2
    C_s = K["k3"] * n * w_m ** 2 + K["c_turn"] * n * davg / s_m
    C_p = K["k7"] * l * w_m
    C_tot = C_p + 2 * C_s
    R_p = K["k6"] / (l * w_m)
    rho = omega * L / R_s
    gamma_g = omega ** 2 * L * C_tot / 2
    delta = R_s ** 2 * C_tot / 2 / L
    Q = (1 - delta - gamma_g) * rho * 2 * R_p / (2 * R_p + (rho ** 2 + 1) * R_s)
    disc = 2 / (L * C_tot) - R_s ** 2 / L ** 2
    fsr = math.sqrt(disc) / (2 * math.pi) if disc > 0 else float("nan")
    return dict(L=L, C_s=C_s, C_p=C_p, C_tot=C_tot, Q=Q, fsr=fsr, d_avg=davg)


def sample(n_target, seed, s_min=S_MIN, s_max=S_MAX, pool_factor=40):
    """LHS over (n, w, s, d_avg); keep feasible (drawable + area-capped), up to n_target.
    s_min/s_max let a follow-up sweep cover a wider spacing band (the GP designs at s=14-17 um,
    outside the original s in [2,7]); the manifests share a schema so the sets COMBINE."""
    pool = max(n_target * pool_factor, 400)
    u = qmc.LatinHypercube(d=4, seed=seed).random(pool)
    nn = np.clip((N_MIN + np.floor(u[:, 0] * (N_MAX - N_MIN + 1))).astype(int), N_MIN, N_MAX)
    ww = W_MIN + u[:, 1] * (W_MAX - W_MIN)
    ss = s_min + u[:, 2] * (s_max - s_min)
    da = DAVG_MIN + u[:, 3] * (DAVG_MAX - DAVG_MIN)
    keep = []
    for n, w, s, davg in zip(nn, ww, ss, da):
        Wrad = n * w + (n - 1) * s
        d_out = davg + Wrad
        d_in = davg - Wrad
        e = w + s + (w + 2 * s) / (1 + RT2)
        if d_in >= e + 2 * w and d_out <= DOUT_CAP:      # drawable AND not huge
            keep.append((int(n), float(w), float(s), float(d_out)))
        if len(keep) >= n_target:
            break
    return keep


def _build_worker(sol, out_dir, q, geom_only=False):
    try:
        ind_name, sim_path = gsp.build_one(sol, out_dir, geom_only=geom_only)
        q.put(("ok", ind_name, sim_path))
    except Exception as e:
        q.put(("error", str(e)[:55], ""))


def build_with_timeout(sol, out_dir, timeout, geom_only=False):
    """Build one model in a SUBPROCESS so a gmsh HANG (a C-level infinite loop, which no
    try/except can catch) can be killed. Returns (ind_name, sim_path) on success, else
    (None, reason). The half-built dir of a killed model has no config.json, so it is
    naturally ignored by the array and by manifest_from_dirs.py."""
    q = mp.Queue()
    proc = mp.Process(target=_build_worker, args=(sol, out_dir, q, geom_only))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate(); proc.join()
        return None, "TIMEOUT (gmsh hang)"
    try:
        status, a, b = q.get_nowait()
    except Exception:
        return None, "no result (crashed)"
    return (a, b) if status == "ok" else (None, a)


def _write_manifest(out_dir, rows):
    # Write to a temp file then atomically rename, so a SLURM kill mid-write can never leave
    # a half-written manifest (the array may read it the instant generation ends, afterany).
    tmp = out_dir / "manifest.csv.tmp"
    with open(tmp, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        wtr.writeheader(); wtr.writerows(rows)
    tmp.replace(out_dir / "manifest.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="target number of feasible models")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--s-min", type=float, default=S_MIN, help="min spacing [um]")
    ap.add_argument("--s-max", type=float, default=S_MAX, help="max spacing [um]")
    ap.add_argument("--out", default="lhs", help="output dir under scripts/.out/")
    ap.add_argument("--geom-only", action="store_true",
                    help="write only the *_forEM.gds per coil (skip the Palace mesh build) "
                         "-- for the rapidfem LHS, which meshes itself. Fast (no gmsh).")
    ap.add_argument("--dry-run", action="store_true", help="print sampling stats, build nothing")
    a = ap.parse_args()

    pts = sample(a.n, a.seed, s_min=a.s_min, s_max=a.s_max)
    arr = np.array([(n, w, s, do) for n, w, s, do in pts], float)
    ln, ld = np.log(arr[:, 0]), np.log([predict(n, w*1e-6, s*1e-6, do*1e-6)["d_avg"]*1e6
                                        for n, w, s, do in pts])
    corr = float(np.corrcoef(ln, ld)[0, 1])
    print(f"sampled {len(pts)} feasible geometries (target {a.n}, seed {a.seed})")
    print(f"  n in [{arr[:,0].min():.0f},{arr[:,0].max():.0f}]  w [{arr[:,1].min():.1f},{arr[:,1].max():.1f}]  "
          f"s [{arr[:,2].min():.1f},{arr[:,2].max():.1f}]  d_out [{arr[:,3].min():.0f},{arr[:,3].max():.0f}] um")
    print(f"  corr(log n, log d_avg) = {corr:+.3f}   (cturn locus was -0.83; near 0 = decorrelated)")
    if a.dry_run:
        return

    OUT = REPO / "scripts" / ".out" / a.out
    OUT.mkdir(parents=True, exist_ok=True)
    gsp.FSWEEP = True
    gsp.FSWEEP_STOP = BAND_STOP
    print(f"band 0.1-{BAND_STOP/1e9:g} GHz, gamma={GAMMA}  ->  {OUT.relative_to(REPO)}\n")

    rows = []
    skipped = 0
    for n, w, s, d_out in pts:
        p = predict(n, w * 1e-6, s * 1e-6, d_out * 1e-6)
        sol = SimpleNamespace(n=n, w=w * 1e-6, s=s * 1e-6, d_out=d_out * 1e-6, L=p["L"])
        # Build in a subprocess with a hard timeout: a single un-meshable geometry (gmsh
        # "overlapping facets" exception) OR a gmsh HANG (uncatchable C-level infinite loop)
        # is killed and skipped instead of crashing or freezing the whole sweep.
        ind_name, info = build_with_timeout(sol, OUT, PER_MODEL_TIMEOUT, geom_only=a.geom_only)
        if ind_name is None:
            skipped += 1
            print(f"  SKIP N={n} w={w:5.1f} s={s:4.1f} d_out={d_out:5.0f}um  ({info})")
            continue
        model_dir = str(Path(info).resolve().relative_to(REPO))
        idx = len(rows) + 1
        rows.append(dict(idx=idx, target_nH=round(p["L"] * 1e9), N=n,
                         w_um=round(w, 4), s_um=round(s, 4), d_out_um=round(d_out, 3),
                         d_avg_um=round(p["d_avg"] * 1e6, 3),
                         pred_L_nH=round(p["L"] * 1e9, 4), pred_Cs_fF=round(p["C_s"] * 1e15, 3),
                         pred_Cp_fF=round(p["C_p"] * 1e15, 3), pred_Ctot_fF=round(p["C_tot"] * 1e15, 3),
                         pred_Q=round(p["Q"], 3), pred_SRF_GHz=round(p["fsr"] / 1e9, 4),
                         model_dir=model_dir))
        # Incremental write: rewrite the manifest after EVERY model so a SLURM TIMEOUT or
        # crash leaves a complete, usable manifest of everything built so far (no salvage run).
        _write_manifest(OUT, rows)
        print(f"  [{idx:3d}] N={n} w={w:5.1f} s={s:4.1f} d_out={d_out:5.0f}um  "
              f"L={p['L']*1e9:5.2f}nH Cs={p['C_s']*1e15:5.1f}fF SRF={p['fsr']/1e9:4.2f}GHz -> {ind_name}")

    if not rows:
        print("No models built (all geometries failed)."); return
    print(f"\nWrote {len(rows)} models + manifest ({skipped} skipped): {OUT/'manifest.csv'}")


if __name__ == "__main__":
    main()
