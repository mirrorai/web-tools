#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

EXP_NAME=$1

# create log directory
LOG_DIR="exps/${EXP_NAME}/logs"
mkdir -p $LOG_DIR

# create output directory
OUT_DIR="exps/${EXP_NAME}/output"
mkdir -p $OUT_DIR

LOG="exps/${EXP_NAME}/logs/train_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

python solve.py --exp_dir exps/${EXP_NAME}

HTML_OUTPUT="exps/${EXP_NAME}/output/train_loss_`date +'%Y-%m-%d_%H-%M-%S'`.html"
python plot_train_curve.py --log ${LOG} --output ${HTML_OUTPUT}