#!/bin/bash
#BSUB -J TR_A_6_24
#BSUB -W 600
#BSUB -n 2
#BSUB -o Out_Files/tree.%J.out
#BSUB -e Error_Files/tree.%J.err
#BSUB -R span[hosts=1]
#BSUB -R "rusage[mem=2GB]"


echo "TREE: $SAMPLE"

/usr/local/usrapps/floreslab/TDA_venv3/bin/python aabc_run_tree.py $SAMPLE



