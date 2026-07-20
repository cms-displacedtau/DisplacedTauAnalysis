#!/bin/bash

SCORE_THRESH="0p99"

SIGDIR=datacards/SR_score_wUncertainties_${SCORE_THRESH}
QCD=datacards/QCD/datacard_QCD_CR.txt
TT=datacards/TT/datacard_TT_CR.txt
W=datacards/W/datacard_W_CR.txt

#Step 1: Combine datacards
for MASS in 100 200 300 400 500; do
  for CTAU in 1 5 10 50 100 1000; do
#for MASS in 500; do
#  for CTAU in 50; do
    SIGNAL="Stau_${MASS}_${CTAU}mm"

    echo "Working on ${SIGNAL}"


    combineCards.py signal_region=$SIGDIR/Stau_${MASS}_${CTAU}mm/datacard_${SIGNAL}.txt \
    qcd_cr=$QCD \
    tt_cr=$TT \
    w_cr=$W \
    > $SIGDIR/Stau_${MASS}_${CTAU}mm/combined_${SIGNAL}.txt

    combineCards.py $SIGDIR/Stau_${MASS}_${CTAU}mm/combined_${SIGNAL}.txt -S > $SIGDIR/Stau_${MASS}_${CTAU}mm/combined_shape_${SIGNAL}.txt

    ## This is to deal with the combined shape datacards
    ## Please make sure these are the correct lines you're replacing! 
    sed -i "26c\\rate_QCD      rateParam * QCD 1.0 [0,5]" "$SIGDIR/Stau_${MASS}_${CTAU}mm/combined_shape_${SIGNAL}.txt"
    sed -i "27c\\rate_Top      rateParam * Top 1.0 [0,5]" "$SIGDIR/Stau_${MASS}_${CTAU}mm/combined_shape_${SIGNAL}.txt"
    sed -i "28c\\rate_WJets      rateParam * WJets 1.0 [0,5]" "$SIGDIR/Stau_${MASS}_${CTAU}mm/combined_shape_${SIGNAL}.txt"
    ## This line deletes lines 31-39 of the file given so we don't have extra copies of rate param
    sed -i "29,38d" "$SIGDIR/Stau_${MASS}_${CTAU}mm/combined_shape_${SIGNAL}.txt"


    text2workspace.py $SIGDIR/Stau_${MASS}_${CTAU}mm/combined_shape_${SIGNAL}.txt -m ${MASS}\
    -o workspace_w_masks_${SIGNAL}_${SCORE_THRESH}.root \
    --channel-masks

    combine -M FitDiagnostics workspace_w_masks_${SIGNAL}_${SCORE_THRESH}.root -m ${MASS} \
    --setParameters mask_ch1_signal_region=1 \
    --saveShapes \
    --plots \
    -n .FitDiagnostics_${SIGNAL}_${SCORE_THRESH} \
    --rMin -1 --rMax 2 \
    --cminDefaultMinimizerStrategy 0 \
    #-v 2
    
    combine -M MultiDimFit workspace_w_masks_${SIGNAL}_${SCORE_THRESH}.root -m ${MASS} \
    --setParameters mask_ch1_signal_region=1,r=0 \
    --freezeParameters r \
    --algo none \
    --saveWorkspace \
    --X-rtd MINIMIZER_no_analytic \
    --cminDefaultMinimizerStrategy 1 \
    -n .CRonlyFit_${SIGNAL}_${SCORE_THRESH}

    combine -M GenerateOnly higgsCombine.CRonlyFit_${SIGNAL}_${SCORE_THRESH}.MultiDimFit.mH${MASS}.root \
    --snapshotName MultiDimFit \
    --setParameters mask_ch1_signal_region=0,r=0 \
    --freezeParameters r     -t -1     --saveToys \
    --X-rtd MINIMIZER_no_analytic \
    -m ${MASS} \
    -n .PostFitAsimov_${SIGNAL}_${SCORE_THRESH}

    combine -M AsymptoticLimits \
    higgsCombine.CRonlyFit_${SIGNAL}_${SCORE_THRESH}.MultiDimFit.mH${MASS}.root \
    --snapshotName MultiDimFit \
    --setParameters mask_ch1_signal_region=0 \
    --run blind \
    --X-rtd MINIMIZER_no_analytic \
    --cminDefaultMinimizerStrategy 1 \
    -m ${MASS} -n .CRconstrained_${SIGNAL}_${SCORE_THRESH}


    #for QUANTILE in 0.025 0.16 0.5 0.84 0.975; do
    #  combine -M HybridNew  higgsCombine.CRonlyFit_${SIGNAL}_${SCORE_THRESH}.MultiDimFit.mH${MASS}.root \
    #  --snapshotName MultiDimFit \
    #  --setParameters mask_ch1_signal_region=0 \
    #  --LHCmode LHC-limits \
    #  --X-rtd MINIMIZER_no_analytic \
    #  -T 50 --fork 4 \
    #  --rMin 0 --rMax 2 \
    #  --rAbsAcc 0.001 --rRelAcc 0.005 \
    #  --expectedFromGrid ${QUANTILE} \
    #  -m ${MASS} -n .HybridNew_${SIGNAL}_${SCORE_THRESH}
    #done  
  done
done

