#!/bin/bash

# Image and model names
TEST_IMG=ADE_val_00001519.jpg
#MODEL_NAME=ade20k-resnet50dilated-ppm_deepsup
MODEL_NAME=ade20k-mobilenetv2dilated-c1_deepsup
MODEL_PATH=ckpt/$MODEL_NAME
RESULT_PATH=./

ENCODER=$MODEL_NAME/encoder_epoch_20.pth
DECODER=$MODEL_NAME/decoder_epoch_20.pth

# Download model weights and image
if [ ! -e $MODEL_PATH ]; then
  mkdir -p $MODEL_PATH
fi
if [ ! -e $ENCODER ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$ENCODER
fi
if [ ! -e $DECODER ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$DECODER
fi
if [ ! -e $TEST_IMG ]; then
  wget -P $RESULT_PATH http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016/images/validation/$TEST_IMG
fi

if [ -z "$DOWNLOAD_ONLY" ]
then