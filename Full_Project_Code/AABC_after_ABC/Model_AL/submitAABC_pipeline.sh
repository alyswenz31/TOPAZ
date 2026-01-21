#!/bin/bash
#BSUB -J AP_A_6_24
#BSUB -W 60   # enough for all steps, adjust as needed
#BSUB -n 1
#BSUB -o out.%J
#BSUB -e err.%J
#BSUB -R "rusage[mem=2GB]"

source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/floreslab/TDA_venv3

echo "Running Step 3 AABC Multiple"
python ABC_S03_crockerplot_distance_aabc_multiple.py

echo "Running Step 4"
python ABC_S04_p01_tolerance.py

echo "Running Step 5"
python ABC_S05_process_sim.py

echo "Running Step 6"
python ABC_S06_process_crocker.py

echo "Running Step BIC"
python bic_final.py

conda deactivate

echo "Pipeline completed"
