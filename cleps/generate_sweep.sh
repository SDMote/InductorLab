#!/bin/bash
# Generate the min..max nH design sweep's Palace models on CLEPS. GP = spiral_cturn.design
# (full Mohan d_out form + fringing C_p, canonical coefficients_sym_sg13g2.py); each coil is
# built in the chosen Palace fixture. Robust build (subprocess timeout + incremental manifest).
# Args:  min max freqGHz srfGHz out_dir fixture[diffpec|backside|grounded] [losslesssub]
# Submit from the repo root, then solve with cleps/run_sweep_array.sh <out_dir>:
#     sbatch cleps/generate_sweep.sh 1 20 2.5 7.0 sweep_srf7 diffpec       # SRF-constrained, true BSG
#     sbatch cleps/generate_sweep.sh 1 20 2.5 2.5 sweep_maxq diffpec       # SRF released
#     sbatch cleps/generate_sweep.sh 12 14 2.5 2.5 sweep_sig0 diffpec losslesssub   # control

#SBATCH --job-name=ind-sweep-gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu_homogen
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/sweep_gen_%j.out
#SBATCH --error=logs/sweep_gen_%j.err

set -x
MIN="${1:-1}"; MAX="${2:-20}"; FREQ="${3:-2.5}"; SRF="${4:-7.0}"; OUT="${5:-sweep}"
FIX=""
case "${6:-}" in
  diffpec)  FIX="--diffpec" ;;
  backside) FIX="--backside" ;;
  grounded) FIX="--grounded" ;;
esac
[ "${7:-}" = "losslesssub" ] && FIX="${FIX} --losslesssub"

module purge
module load gnu13
source /home/$USER/.bashrc
conda activate inductor_lab

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
python3 scripts/palace/generate_sweep_palace.py --min "${MIN}" --max "${MAX}" --freq "${FREQ}" \
    --srf "${SRF}" --out "${OUT}" ${FIX}

echo "### sweep generation done: scripts/.out/${OUT}/manifest.csv ###"
