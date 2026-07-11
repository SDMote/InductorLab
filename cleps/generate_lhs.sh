#!/bin/bash
# Generate the Latin-hypercube geometry sweep's Palace models on CLEPS. Fixed seed/count
# so the manifest is reproducible (local analysis reads the same rows CLEPS produced).
# Args: N seed s_min s_max out_dir.  Submit from the repo root:
#     sbatch cleps/generate_lhs.sh                          # 120, seed0, s[2,7], -> lhs
#     sbatch cleps/generate_lhs.sh 100 1 7 30 lhs_wides     # wider-s follow-up (combines)
# Single-process gmsh meshing; coils run up to ~800 um so allow time.

#SBATCH --job-name=ind-lhs-gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu_homogen
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=logs/lhs_gen_%j.out
#SBATCH --error=logs/lhs_gen_%j.err

set -x
NMODELS="${1:-120}"
SEED="${2:-0}"
SMIN="${3:-2}"
SMAX="${4:-7}"
OUT="${5:-lhs}"
# 6th arg: pass "diffpec" for the DIFFERENTIAL 1-port + PEC-ground backside fixture.
DIFFPEC=""; [ "${6:-}" = "diffpec" ] && DIFFPEC="--diffpec"

module purge
module load gnu13
source /home/$USER/.bashrc
conda activate inductor_lab

cd "${SLURM_SUBMIT_DIR}"
python3 scripts/palace/generate_lhs_sweep.py --n "${NMODELS}" --seed "${SEED}" \
    --s-min "${SMIN}" --s-max "${SMAX}" --out "${OUT}" ${DIFFPEC}

echo "### lhs generation done: scripts/.out/${OUT}/manifest.csv ###"
