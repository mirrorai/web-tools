#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

EXP_NAME=$1
LAST_EPOCH=$2

# create log directory
LOG_DIR="exps/${EXP_NAME}/logs"
mkdir -p $LOG_DIR

# create output directory
OUT_DIR="exps/${EXP_NAME}/output"
mkdir -p $OUT_DIR

LOG="exps/${EXP_NAME}/logs/train_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

if ! [ -n "$LAST_EPOCH" ];
then
   LAST_EPOCH=`find exps/${EXP_NAME}/snapshots -type f -ipath "*${LAST_EPOCH}*.params" -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" " | xargs basename | tr -cd '[[:digit:]]'`
fi

python solve.py --exp_dir exps/${EXP_NAME} --resume_model ${LAST_EPOCH}

HTML_OUTPUT="exps/${EXP_NAME}/output/train_loss_`date +'%Y-%m-%d_%H-%M-%S'`.html"
python plot_train_curve.py --log ${LOG} --output ${HTML_OUTPUT}