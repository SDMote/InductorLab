#!/bin/bash
# SLURM ARRAY: solve a Palace inductor sweep built by cleps/generate_sweep.sh, one coil per
# task. Each model is whatever fixture the generate used (--diffpec => differential 1-port .s1p;
# --backside => plug 2-port .s2p). Build first with cleps/generate_sweep.sh <...> <out_dir>.
#
# Each task = manifest row idx (col 1) -> model_dir (last col); rows past the last skip
# cleanly. Writes the adaptive 0.1-15 GHz Touchstone into <model_dir>/output via combine_snp.
# Arg: sweep dir under scripts/.out/. Submit from the repo root (= SLURM_SUBMIT_DIR):
#     sbatch cleps/run_sweep_array.sh sweep_srf7                     # solve that sweep dir
#     sbatch --dependency=afterok:$gid cleps/run_sweep_array.sh sweep_srf7   # after generate
# Pull back (.s1p for diffpec, .s2p for plug):
#     ssh cleps "cd ~/InductorLab && find scripts/.out/<out_dir> \
#       \( -iname '*.s1p' -o -iname '*.s2p' -o -name 'manifest.csv' \) | tar czf - -T -" | tar xzf - -C .

#SBATCH --job-name=palace-sweep
#SBATCH --array=1-20%6
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=simon.jaramillo-haddad@inria.fr,sjaramillo5@uc.cl
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu_homogen
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rfc_%A_%a.out
#SBATCH --error=logs/rfc_%A_%a.err

set -x
SWEEP="${1:-sweep}"
MANIFEST="${SLURM_SUBMIT_DIR}/scripts/.out/${SWEEP}/manifest.csv"
if [ ! -f "${MANIFEST}" ]; then echo "manifest not found: ${MANIFEST}"; exit 1; fi

MODEL_DIR=$(awk -F, -v id="${SLURM_ARRAY_TASK_ID}" 'NR>1 && $1==id {print $NF}' "${MANIFEST}" | tr -d '\r')
if [ -z "${MODEL_DIR}" ]; then
    echo "### task ${SLURM_ARRAY_TASK_ID}: no manifest row (past last model) -- skip ###"; exit 0
fi
echo "### sweep task ${SLURM_ARRAY_TASK_ID} -> ${MODEL_DIR} ###"

module purge
module load gnu13 openmpi5
source /home/$USER/.bashrc
conda activate inductor_lab
export PATH="${SLURM_SUBMIT_DIR}/cleps:$PATH"

mkdir -p "${SLURM_SUBMIT_DIR}/logs"
cd "${SLURM_SUBMIT_DIR}/${MODEL_DIR}" || { echo "cd failed: ${MODEL_DIR}"; exit 1; }
[ -f config.json ] || { echo "no config.json in $(pwd)"; exit 1; }
run_palace config.json
combine_snp

echo "### sweep task ${SLURM_ARRAY_TASK_ID} done: ${SLURM_SUBMIT_DIR}/${MODEL_DIR} ###"
