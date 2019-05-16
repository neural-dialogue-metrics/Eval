#!/usr/bin/env bash

SERBAN_IMAGE=ufoym/deepo:theano-py36-cu90
SERBAN_ROOT=/home/cgsdfc/deployment/Models/HRED-VHRED
GPU_INDEX=0

MODEL_PREFIX={model_prefix}
TEST_PATH={test_path}
SAVE_DIR={save_dir}

BS=80

docker run --rm -it --runtime nvidia \
    --name {name} \
    -v $HOME:$HOME \
    -w $SERBAN_ROOT \
    -e PYTHONPATH=$SERBAN_ROOT \
    -e THEANO_FLAGS=device=cuda$GPU_INDEX \
    $SERBAN_IMAGE \
    python bin/evaluate.py $MODEL_PREFIX \
        --test-path $TEST_PATH \
        --save-dir $SAVE_DIR \
        {remove_stopwords} \
        --bs $BS
