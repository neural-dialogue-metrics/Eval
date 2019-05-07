#!/usr/bin/env bash

GPU=0
SAVE_DIR={save_dir}
SERBAN_ROOT=/home/cgsdfc/deployment/Models/HRED-VHRED
SERBAN_IMAGE=ufoym/deepo:theano-py36-cu90

docker run --rm -it --runtime nvidia \
    --name {name} \
    -v $HOME:$HOME \
    -w $SERBAN_ROOT \
    -e PYTHONPATH=$SERBAN_ROOT \
    -e THEANO_FLAGS=device=cuda$GPU \
    $SERBAN_IMAGE \
    python bin/train.py {prototype} \
    --save-dir $SAVE_DIR \
    --prefix {model_prefix} \
    --auto_restart
