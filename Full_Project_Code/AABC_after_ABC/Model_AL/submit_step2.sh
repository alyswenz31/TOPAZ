#!/bin/bash
#BSUB -J 2_A_06_24
#BSUB -W 7200
#BSUB -n 4
#BSUB -o Out_Files/step2.%J.out
#BSUB -e Error_Files/step2.%J.err
#BSUB -R span[hosts=1]

/usr/local/usrapps/floreslab/TDA_venv3/bin/python ABC_S02_crockerplot_save.py 


