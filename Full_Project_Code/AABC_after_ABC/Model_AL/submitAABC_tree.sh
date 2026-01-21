#!/bin/bash
#BSUB -J AA_A_6_24
#BSUB -W 60
#BSUB -n 4
#BSUB -o out.%J
#BSUB -e err.%J
#BSUB -R span[hosts=1]
#BSUB -R "rusage[mem=60GB]"

source ~/.bashrc
module load conda   #delete if using system Python 2
conda activate /usr/local/usrapps/floreslab/TDA_venv3
python aabc_run_tree.py #5-20ish minutes
conda deactivate

