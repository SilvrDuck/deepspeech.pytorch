#!/bin/bash


if [ $# -eq 0 ]
  then
    echo "Error: give model path as argument."
    exit 1
fi

MODEL_PATH=$1
MODEL=$(basename $MODEL_PATH)

SPLITS="./data/LogiNonNative_dataset/splits/"
LOG="./logs/test_"${MODEL}

LM=$2
ARGS="--cuda --verbose --decoder beam --lm-path $LM"

printf "LogiNonNative_$LM\n"
A=$(python ./test.py --model-path ${MODEL_PATH} --test-manifest ${SPLITS}native.manifest ${ARGS})

printf "$A"
