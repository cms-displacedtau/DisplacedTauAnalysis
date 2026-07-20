#!/bin/bash -x


ARGS="$3"

if [ -n "$SUFFIX" ]; then
    SUFFIX="_${SUFFIX}"
fi



#DIR=~/mnt/desy_dust/sobhatta/work/LongLivedStaus/LLStaus_Run2/Limits/results/limits${SUFFIX}/llstau_${TYPE}/channels_all/eras_all
#DIR=~/mnt/desy_dust/sobhatta/work/LongLivedStaus/LLStaus_Run2/Limits/results/limits_test-distau-syst-20percent/llstau_${TYPE}/channels_all/eras_all
#DIR=~/mnt/desy_dust/sobhatta/work/LongLivedStaus/LLStaus_Run2/Limits/results/limits_dxy-gt-0p2_8-bins/llstau_${TYPE}/channels_all/eras_all
#DIR=results/limits${SUFFIX}/llstau_${TYPE}/channels_all/eras_all
DIR=plots_0p99

XSECFILE=crosssections_stau_hepi-fast.csv

python3 plot_limits_new.py \
--jsons limits_Stau_*_*mm.json \
--xsecfile $XSECFILE \
--cmsextratext "Simulation" \
--outdir $DIR \
--blind
${ARGS}
