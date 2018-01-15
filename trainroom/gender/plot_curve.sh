#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

EXP_NAME=$1

LOG=`find exps/${EXP_NAME}/logs -type f -ipath '*train*.txt*' -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2 -d" "`
echo ${LOG}

DATE_STR="`date +'%Y-%m-%d_%H-%M-%S'`"

HTML_OUTPUT="exps/${EXP_NAME}/output/train_loss_${DATE_STR}.html"

python plot_train_curve.py --log ${LOG} --output ${HTML_OUTPUT}

HTML_OUTPUT_TEST="exps/${EXP_NAME}/output/test_acc_${DATE_STR}.html"

python plot_val_curve.py --log ${LOG} --output ${HTML_OUTPUT_TEST}