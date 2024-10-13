@echo off

:: Set the working directory
cd <project PYTHONPATH>

:: Activate the conda environment
call conda activate <your environment>

:: Set necessary environment variables
set PYTHONPATH=%CD%
set CUDA_VISIBLE_DEVICES=0
python runnables/train_CFPnet.py -m +dataset=mimic3_real +backbone=CFPnet +backbone/CFPnet_hparams/mimic3_real=diastolic_blood_pressure exp.seed=10,100,1000,10000,100000
echo All runs complete