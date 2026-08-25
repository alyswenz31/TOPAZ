#!/bin/bash
#BSUB -J 1_D_06_24           
#BSUB -W 4320                  
#BSUB -n 4                     
#BSUB -o Out_Files/step1.%J.out
#BSUB -e Error_Files/step1.%J.err            
#BSUB -R span[hosts=1]         

/usr/local/usrapps/floreslab/TDA_venv3/bin/python ABC_S01_samples.py 


