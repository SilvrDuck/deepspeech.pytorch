#!/bin/bash

EXP_NAME="First_large_scale_accent_classification_test" # no spaces
DEV_OR_TRAIN="train"
EPOCHS='4'
MODEL='mtaccent' # deepspeech or mtaccent

NOW=$(eval date +"%F_%Hh%M_")
ID=$NOW$EXP_NAME

SPLITS="data/CommonVoice_dataset/splits/"

MODELS_PATH="/data/thibault/deepspeech_saves/history/"$ID
mkdir $MODELS_PATH
RUNS_PATH="runs/"$ID
mkdir $RUNS_PATH

echo Starting $DEV_OR_TRAIN training of $MODEL model

time python train.py \
	--model $MODEL \
	--train-manifest ${SPLITS}${DEV_OR_TRAIN}.csv \
	--val-manifest ${SPLITS}test.csv \
	--sample-rate 16000 \
	--batch-size 20 \
	--window-size .02 \
	--window-stride .01 \
	--window hamming \
	--hidden-size 800 \
	--hidden-layers 5 \
    --side-hidden-layers 4 \
    --side-hidden-size 800 \
    --side-rnn-type gru \
    --shared-layers 2 \
    --mixing-coef .5 \
	--rnn-type gru \
	--epochs $EPOCHS \
	--lr 3e-4 \
	--momentum 0.9 \
	--max-norm 400 \
	--learning-anneal 1.1 \
	--checkpoint \
	--tensorboard \
	--log-dir $RUNS_PATH \
	--visdom \
	--log-params \
	--save-folder $MODELS_PATH \
	--model-path models/best/${ID}.pth \
	--cuda \
	--augment \
	--id $ID
