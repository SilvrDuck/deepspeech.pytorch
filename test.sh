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
A=$(python ./test.py --model-path ${MODEL_PATH} --test-manifest ${SPLITS}dev.csv ${ARGS})
printf "$A\n"
printf "Testing test\n"
B=$(python ./test.py --model-path ${MODEL_PATH} --test-manifest ${SPLITS}test.csv ${ARGS})
printf "$B\n"
printf "Testing test-in\n"
C=$(python ./test.py --model-path ${MODEL_PATH} --test-manifest ${SPLITS}testin.csv ${ARGS})
printf "$C\n"
printf "Testing test-nz\n"
D=$(python ./test.py --model-path ${MODEL_PATH} --test-manifest ${SPLITS}testnz.csv ${ARGS})
printf "$D\n"

printf "Dev\t$A\nTest\t$B\nTest-in\t$C\nTest-nz\t$D\n" > ${LOG}.log