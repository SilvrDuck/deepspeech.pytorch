#!/bin/bash

EXP_NAME="Only_three_epochs" # no spaces
DEV_OR_TRAIN="dev"

NOW=$(eval date +"%F_%Hh%M_")
ID=$NOW$EXP_NAME

SPLITS="data/CommonVoice_dataset/splits/"

mkdir models/saved/${NOW}

python train.py \
	--train-manifest ${SPLITS}${DEV_OR_TRAIN}.csv \
	--val-manifest ${SPLITS}test.csv \
	--sample-rate 16000 \
	--batch-size 20 \
	--window-size .02 \
	--window-stride .01 \
	--window hamming \
	--hidden-size 800 \
	--hidden-layers 5 \
	--rnn-type gru \
	--epochs 3 \
	--lr 3e-4 \
	--momentum 0.9 \
	--max-norm 400 \
	--learning-anneal 1.1 \
	--checkpoint \
	--tensorboard \
	--visdom \
	--log-dir runs/ \
	--log-params \
	--save-folder models/saved/${ID}/ \
	--model-path models/best/${ID}.pth \
	--cuda \
	--augment \
	--id $ID
