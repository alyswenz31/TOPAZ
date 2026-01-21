#!/bin/bash
#BSUB -J PARS
#BSUB -W 200  
#BSUB -n 1
#BSUB -o out.%J
#BSUB -e err.%J

source ~/.bashrc
module load conda   #delete if using system Python 2
conda activate /usr/local/usrapps/floreslab/TDA_venv3
python get_pars_n_crockers.py 
conda deactivate

