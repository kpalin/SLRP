#!/bin/bash

# For debugging
#set -o verbose

# Die on unset variables
set -o nounset
# Die on errors
set -o errexit
# Die if any part of a pipe fails
set -o pipefail

OUTBASE="plink"

usage()  {
    echo -e "usage:\n$0 -b/-f basename [-o outbase] [-- plinkoptions]

-f inbase   Base name for map/ped files
-t inbase   Base name for tfam/tped files
-b inbase   Base name for bim/bed/fam files
-o outbase  Base name for output and temporary files files (default ${OUTBASE})
-h          Show this message and exit.

The rest of the command line is given to plink for possible data filtering.
" >&2
    exit 1

}


while getopts  "h:f:t:b:o:-" flag
do
  case "$flag" in 
      o)
      OUTBASE="$OPTARG"
      ;;
      f)
      INBASE="$OPTARG"
      INCMD="--file"
      ;;
      t)
      INBASE="$OPTARG"
      INCMD="--tfile"
      ;;
      b)
      INBASE="$OPTARG"
      INCMD="--bfile"
      ;;
      -)
      break
      ;;
      *)
#      usage;
      ;;
  esac
done

shift $((OPTIND-1)); OPTIND=1

trap usage EXIT



test -s ${OUTBASE}.tfam || plink --noweb ${INCMD} ${INBASE}  --out ${OUTBASE} $@  --transpose  --recode12

awk --assign OUTBASE=${OUTBASE} '{
outStr=$4 " ";
for(i=5;i<=NF;i+=2) {
j=i+1;
    if($i=="0") outStr=outStr "33"
    else if($i==$j) outStr=outStr $i-1 $i-1
    else outStr=outStr "22";
} print outStr >OUTBASE ".chr" $1 ".fad";
}' ${OUTBASE}.tped

trap EXIT