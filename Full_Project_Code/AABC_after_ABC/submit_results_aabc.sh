#!/bin/bash
#BSUB -J RESULT
#BSUB -W 30  
#BSUB -n 1
#BSUB -o out.%J
#BSUB -e err.%J

source ~/.bashrc
module load conda   #delete if using system Python 2
conda activate /usr/local/usrapps/floreslab/TDA_venv3
python get_results_aabc.py 
conda deactivate

