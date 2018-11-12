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

printf "Testing dev\n"
A=$(python ./test.py --model-path ${MODEL_PATH} --test-manifest ${SPLITS}dev.csv --cuda)
printf "$A\n"
printf "Testing test\n"
B=$(python ./test.py --model-path ${MODEL_PATH} --test-manifest ${SPLITS}test.csv --cuda)
printf "$B\n"
printf "Testing test-in\n"
C=$(python ./test.py --model-path ${MODEL_PATH} --test-manifest ${SPLITS}testin.csv --cuda)
printf "$C\n"
printf "Testing test-nz\n"
D=$(python ./test.py --model-path ${MODEL_PATH} --test-manifest ${SPLITS}testnz.csv --cuda)
printf "$D\n"

printf "Dev\t$A\nTest\t$B\nTest-in\t$C\nTest-nz\t$D\n" > ${LOG}.log
