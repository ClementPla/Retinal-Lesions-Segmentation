#!/usr/bin/env bash

#
#SBATCH --job-name=FundusSegmentation
#SBATCH --output=log_task.txt
#SBATCH --gres=gpu:a6000:1
#SBATCH --partition=gpu

ssh -N -L localhost:5010:localhost:5010 clement@m3202-10.demdgi.polymtl.ca &
pid=$!

#Load le module anaconda
#source /etc/profile.d/modules.sh
module load anaconda3

source activate ~/.conda/envs/torch18

python main.py --config config.yaml --models Unet

kill $pid
