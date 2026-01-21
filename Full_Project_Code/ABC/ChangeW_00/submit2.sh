#!/bin/bash
#BSUB -W 1440
#BSUB -n 1
#BSUB -o out.%J
#BSUB -e err.%J

source ~/.bashrc
module load conda   #delete if using system Python 2
conda activate /usr/local/usrapps/floreslab/TDA_venv3
#python ABC_S01_samples.py #1 day took 6 hours so give 7 next time
python ABC_S02_crockerplot_save.py #2 day took 11 hours so give 12 next time
#python ABC_S03_crockerplot_distance.py #12 hours
#python ABC_S04_p01_tolerance.py #12 hours
#python ABC_S05_process_sim.py #unsure
#python ABC_S06_process_crocker.py #unsure
conda deactivate

