#!/bin/bash

## Parameters

EXP_NAME="__tmp__" # no spaces
METHOD="mini" # train, test or mini

MODEL='mtaccent' # deepspeech or mtaccent

EPOCHS='50'
LR='3e-3'

HIDDEN_LAYERS='4'
HIDDEN_SIZE='600'

SIDE_HIDDEN_LAYERS='2'
SHARED_LAYERS='2'

BOTTLENECK_SIZE='100'
MIXING_COEF='.5'

RNN_TYPE='gru'

## Get arguments

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
	--model)
	MODEL="$2"
	shift # past argument
    shift # past value
    ;;
    --exp-name)
	EXP_NAME="$2"
	shift # past argument
    shift # past value
    ;;
	-m|--method)
	METHOD="$2"
	shift # past argument
    shift # past value
    ;;
	--epochs)
	EPOCHS="$2"
	shift # past argument
    shift # past value
    ;;
	--lr)
	LR="$2"
	shift # past argument
    shift # past value
    ;;
	--hidden-layers)
	HIDDEN_LAYERS="$2"
	shift # past argument
    shift # past value
    ;;
	--hidden-size)
	HIDDEN_SIZE="$2"
	shift # past argument
    shift # past value
    ;;
	--side-hidden-layers)
	SIDE_HIDDEN_LAYERS="$2"
	shift # past argument
    shift # past value
    ;;
	--shared-layers)
	SHARED_LAYERS="$2"
	shift # past argument
    shift # past value
    ;;
	--bottleneck-size)
	BOTTLENECK_SIZE="$2"
	shift # past argument
    shift # past value
    ;;
	--mixing-coef)
	MIXING_COEF="$2"
	shift # past argument
    shift # past value
    ;;
	--side-hidden-size)
	SIDE_HIDDEN_SIZE="$2"
	shift # past argument
    shift # past value
    ;;
	--rnn-type)
	RNN_TYPE="$2"
	shift # past argument
    shift # past value
    ;;
	--side-rnn)
	SIDE_RNN_TYPE="$2"
	shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters


SIDE_HIDDEN_SIZE=$HIDDEN_SIZE
SIDE_RNN_TYPE=$RNN_TYPE

## Name

NOW=$(eval date +"%F_%Hh%M_")
MODEL_DEPENDANT=''
if [ "$MODEL" = "mtaccent" ] ; then
	MODEL_DEPENDANT='_slyrs-'${SIDE_HIDDEN_LAYERS}x${SIDE_HIDDEN_SIZE}_shrd-${SHARED_LAYERS}_btnck-${BOTTLENECK_SIZE}_mix-${MIXING_COEF}
fi

ID=${NOW}${EXP_NAME}_model-${MODEL}_ep-${EPOCHS}_lr-${LR}_lyrs-${HIDDEN_LAYERS}x${HIDDEN_SIZE}${MODEL_DEPENDANT}

## Roomkeeping

SPLITS="data/CommonVoice_dataset/splits/"

if [ "$METHOD" = "mini" ] ; then
	MODELS_PATH="saved_models/tmp"
	RUNS_PATH="runs/tmp"
	VALIDATION=${SPLITS}minidev.csv
	CUDA=''
	EPOCHS='2'
	HIDDEN_LAYERS='4'
	HIDDEN_SIZE='80'
	SIDE_HIDDEN_LAYERS='2'
	SHARED_LAYERS='2'
	BOTTLENECK_SIZE='10'
	SIDE_HIDDEN_SIZE=$HIDDEN_SIZE
	BATCH_SIZE='2'
	TENSORBOARD=''
	VISDOM=''
else
	MODELS_PATH="/data/thibault/deepspeech_saves/history/"$ID
	mkdir $MODELS_PATH
	RUNS_PATH="runs/"$ID
	mkdir $RUNS_PATH
	VALIDATION=${SPLITS}dev.csv
	CUDA='--cuda'
	TENSORBOARD='--tensorboard'
	VISDOM='--visdom'
	BATCH_SIZE='20'
fi

## Launching

echo ${METHOD}ing of $MODEL model
echo experiment ID: $ID

time python train.py \
	--model $MODEL \
	--train-manifest ${SPLITS}${METHOD}.csv \
	--val-manifest $VALIDATION \
	--sample-rate 16000 \
	--batch-size $BATCH_SIZE \
	--window-size .02 \
	--window-stride .01 \
	--window hamming \
	--hidden-size $HIDDEN_SIZE \
    --bottleneck-size $BOTTLENECK_SIZE \
    --hidden-layers $HIDDEN_LAYERS \
    --side-hidden-layers $SIDE_HIDDEN_LAYERS \
    --side-hidden-size $SIDE_HIDDEN_SIZE \
    --side-rnn-type $SIDE_RNN_TYPE \
    --shared-layers $SHARED_LAYERS \
    --mixing-coef $MIXING_COEF \
	--rnn-type $RNN_TYPE \
	--epochs $EPOCHS \
	--lr $LR \
	--momentum 0.9 \
	--max-norm 400 \
	--learning-anneal 1.1 \
	--checkpoint $TENSORBOARD $VISDOM \
	--log-dir $RUNS_PATH \
	--log-params \
	--save-folder $MODELS_PATH \
	--model-path saved_models/best/${ID}.pth \
	--augment $CUDA \
	--id $ID
