#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

EXP_NAME=$1
NET_FINAL=$2

# create log directory
LOG_DIR="exps/${EXP_NAME}/logs"
mkdir -p $LOG_DIR

LOG="exps/${EXP_NAME}/logs/test_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

if [ -n "$NET_FINAL" ];
then
   NET_FINAL=`find exps/${EXP_NAME}/snapshots -type f -ipath "*${NET_FINAL}*.params" -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" "`
else
   NET_FINAL=`find exps/${EXP_NAME}/snapshots -type f -ipath '*.params' -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" "`
fi

echo ${NET_FINAL}

python test.py --exp_dir exps/${EXP_NAME} --model_name ${NET_FINAL} --gpus 0
