#!/bin/bash

MODEL="2018-10-15_13h15_Only_three_epochs.pth"

MODEL_PATH="./models/best/"$MODEL
SPLITS="./data/CommonVoice_dataset/splits/"
LOG="./logs/test_"$MODEL

python ./test.py --model-path ${MODEL_PATH} --test-manifest ${SPLITS}dev.csv --cuda > ${LOG}dev.log
python ./test.py --model-path ${MODEL_PATH} --test-manifest ${SPLITS}test.csv --cuda > ${LOG}test.log
python ./test.py --model-path ${MODEL_PATH} --test-manifest ${SPLITS}testin.csv --cuda > ${LOG}testin.log
python ./test.py --model-path ${MODEL_PATH} --test-manifest ${SPLITS}testnz.csv --cuda > ${LOG}testnz.log
