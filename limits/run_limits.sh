#!/bin/bash

SCORE_THRESH="0p99"

# Step 3: Collect all limits into a single JSON summary
#nice -n 10 combineTool.py -M CollectLimits higgsCombine.CRconstrained_Stau_*.*Limits*.* -o limits.json
for MASS in 100 200 300 400 500; do
  for CTAU in 1 5 10 50 100 1000; do
    nice -n 10 combineTool.py -M CollectLimits \
      higgsCombine.CRconstrained_Stau_${MASS}_${CTAU}mm_${SCORE_THRESH}.*Limits*.mH${MASS}.root \
      -o limits_Stau_${MASS}_${CTAU}mm.json
  done
done
