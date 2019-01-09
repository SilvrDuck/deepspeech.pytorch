#!/bin/bash


if [ $# -eq 0 ]
  then
    echo "Error: give model path as argument."
    exit 1
fi

MODEL_PATH=$1
MODEL=$(basename $MODEL_PATH)

SPLITS="./data/Perso_dataset/"
LOG="./logs/test_"${MODEL}

ARGS="--cuda --decoder beam --lm-path data/language_models/cv_train.binary --verbose"

printf "Perso test\n"
A=$(python ./test.py --model-path ${MODEL_PATH} --test-manifest ${SPLITS}manifest.csv ${ARGS})

printf "$A"
