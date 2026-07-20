#!/bin/bash
# ──────────────────────────────────────────────────────────
# Run cutflow script for all 30 Stau (mass, ctau) samples
# ──────────────────────────────────────────────────────────

MASSES=(100 200 300 400 500)
CTAUS=(1 5 10 50 100 1000)


BASE="/eos/uscms/store/user/lpcdisptau/dally/displacedTaus/selected/Summer22_CHS_v19/mutau/v8/PR_score/faster_trial"
SCRIPT="makeCutflow.py"
JSON="../scripts/original_counts.json"
OUTDIR="plots/cutflow"

# Build the sample arguments
SAMPLES=()
for MASS in "${MASSES[@]}"; do
    for CTAU in "${CTAUS[@]}"; do
        TAG="Stau_${MASS}_${CTAU}mm"
        FILE="${BASE}/faster_trial_${TAG}/faster_trial_${TAG}.root"
        SAMPLES+=("-s" "${TAG}:${FILE}")
        python "${SCRIPT}" \
            "${SAMPLES[@]}" \
            -j "${JSON}" \
            -o "${OUTDIR}"
    done
done

echo "Running cutflow for ${#SAMPLES[@]} sample entries ($(( ${#MASSES[@]} * ${#CTAUS[@]} )) samples)..."


echo "Done. Plots saved to ${OUTDIR}/"
