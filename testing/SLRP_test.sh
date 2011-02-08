#!/bin/bash
#BSUB -M 15000000 
#BSUB -q basement
#BSUB -R "select[mem>15000] rusage[mem=15000] span[hosts=1]"
#BSUB -J SLRP_NFBC[1]
#BSUB -o SLRP_NFBC.%I.%J.log
#BSUB -n 6

# For debugging
#set -o verbose

# Die on unset variables
set -o nounset
# Die on errors
set -o errexit
# Die if any part of a pipe fails
set -o pipefail


echo "Long range phase NFBC"

LSB_JOBINDEX=22

INFAD="/lustre/scratch103/sanger/kp5/Suomi/genotypes/NFBC/NFBC.chr${LSB_JOBINDEX}.fad"

GENOMAP="/lustre/scratch103/sanger/kp5/hapmap3/recombRates/genetic_map_chr${LSB_JOBINDEX}_b36.txt"

OUTFAD=`basename ${INFAD} .fad`.hfad
OUTIBD=`basename ${INFAD} .fad`.ibd
OUTLIKE=`basename ${INFAD} .fad`.like


python-2.7 ${HOME}/Kuusamo/scripts/testing/SLRP/SLRP.py -n2  \
    --geneticMap=${GENOMAP} \
    --slice_length=1000 \
    --ibdFile=test_NFBC.chr{LSB_JOBINDEX}.ibd.aibd
    --fadFile=${INFAD} \
    --outFile=${OUTFAD}  \
    --ibdSegCalls=${OUTIBD} \
    --iterations=20 \
    --likeFile=${OUTLIKE} 


