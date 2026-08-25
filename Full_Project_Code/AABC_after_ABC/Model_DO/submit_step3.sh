#!/bin/bash
#BSUB -J S3_D_6_24
#BSUB -W 600
#BSUB -n 2
#BSUB -o Out_Files/step3.%J.out
#BSUB -e Error_Files/step3.%J.err
#BSUB -R "rusage[mem=2GB]"


echo "STEP3: $SAMPLE"

/usr/local/usrapps/floreslab/TDA_venv3/bin/python ABC_S03_crockerplot_distance.py $SAMPLE
