#!/bin/bash


if [ $# -eq 0 ]
  then
    echo "Error: give model path as argument."
    exit 1
fi

MODEL_PATH=$1
MODEL=$(basename $MODEL_PATH)

SPLITS="./data/CommonVoice_dataset/splits/"
LOG="./logs/test_"${MODEL}

ARGS="--decoder beam --lm-path data/language_models/en-70k-0.2.binary"

printf "Testing dev\n"
python ./test.py --model-path ${MODEL_PATH} --test-manifest ${SPLITS}mini.csv ${ARGS}
