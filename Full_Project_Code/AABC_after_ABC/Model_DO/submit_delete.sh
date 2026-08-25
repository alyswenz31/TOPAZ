#!/bin/bash
#BSUB -J DL_D_6_24
#BSUB -o Out_Files/delete.%J.out
#BSUB -e Error_Files/delete.%J.err
#BSUB -W 600
#BSUB -n 2
#BSUB -R "rusage[mem=2GB]"

echo "Deleting sample_aabc_$SAMPLE"

rm -rf sample_aabc_$SAMPLE