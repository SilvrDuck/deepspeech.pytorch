#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "Error, no argument supplied"
    echo "Usage: save_experiment.sh experiment_name pth_file_path"
else
  NOW=$(eval date +"%F_")

  EXP_PATH=/data/thibault/deepspeech_saves/experiments/$NOW$1

  mkdir $EXP_PATH
  cp $2 $EXP_PATH

  PTH_NAME=$(basename -s .pth $2)
  RUN_PATH=runs/$PTH_NAME
  cp -r $RUN_PATH $EXP_PATH
fi
