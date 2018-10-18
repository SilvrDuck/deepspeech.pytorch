#!/bin/bash

EXP_NAME="Baseline_vanilla_deepspeech" # no spaces
DEV_OR_TRAIN="train"
EPOCHS='70'


NOW=$(eval date +"%F_%Hh%M_")
ID=$NOW$EXP_NAME

SPLITS="data/CommonVoice_dataset/splits/"

mkdir models/saved/${NOW}

python train.py \
	--model deepspeech \
	--train-manifest ${SPLITS}${DEV_OR_TRAIN}.csv \
	--val-manifest ${SPLITS}test.csv \
	--sample-rate 100600 \
	--batch-size 20 \
	--window-size .02 \
	--window-stride .01 \
	--window hamming \
	--hidden-size 800 \
	--hidden-layers 5 \
	--rnn-type gru \
	--epochs $EPOCHS \
	--lr 3e-4 \
	--momentum 0.9 \
	--max-norm 400 \
	--learning-anneal 1.1 \
	--checkpoint \
	--tensorboard \
	--log-dir runs/ \
	--visdom \
	--log-params \
	--save-folder models/saved/${ID}/ \
	--model-path models/best/${ID}.pth \
	--cuda \
	--augment \
	--id $ID
