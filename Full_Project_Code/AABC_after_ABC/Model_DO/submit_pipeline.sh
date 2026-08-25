#!/bin/bash
#BSUB -J AP_D_6_24
#BSUB -W 40
#BSUB -n 2
#BSUB -o Out_Files/pipe.%J.out
#BSUB -e Error_Files/pipe.%J.err
#BSUB -R "rusage[mem=2GB]"

/usr/local/usrapps/floreslab/TDA_venv3/bin/python ABC_S04_p01_tolerance_combine.py
/usr/local/usrapps/floreslab/TDA_venv3/bin/python ABC_S05_process_sim.py
/usr/local/usrapps/floreslab/TDA_venv3/bin/python ABC_S06_process_crocker.py
/usr/local/usrapps/floreslab/TDA_venv3/bin/python BIC_and_AIC_calculation.py
/usr/local/usrapps/floreslab/TDA_venv3/bin/python BIC_convergence_test.py

