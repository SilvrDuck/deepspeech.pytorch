#!/bin/bash

EXP_NAME="test_mini" # no spaces
DEV_OR_TRAIN="dev"
EPOCHS='2'


NOW=$(eval date +"%F_%Hh%M_")
ID=$NOW$EXP_NAME

SPLITS="data/CommonVoice_dataset/splits/"

SAVE_PATH="models/saved/"${ID}

mkdir $SAVE_PATH

python train.py \
	--model deepspeech \
	--train-manifest ${SPLITS}mini${DEV_OR_TRAIN}.csv \
	--val-manifest ${SPLITS}minitest.csv \
	--sample-rate 16000 \
	--batch-size 2 \
	--window-size .02 \
	--window-stride .01 \
	--window hamming \
	--hidden-size 100 \
	--hidden-layers 3 \
	--rnn-type gru \
	--epochs $EPOCHS \
	--lr 3e-4 \
	--momentum 0.9 \
	--max-norm 400 \
	--learning-anneal 1.1 \
	--checkpoint \
	--log-dir runs/ \
	--log-params \
	--save-folder $SAVE_PATH \
	--model-path models/best/${ID}.pth \
	--augment \
	--id $ID
