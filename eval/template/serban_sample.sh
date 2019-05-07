#!/usr/bin/env bash

MODEL_PREFIX={model_prefix}
CONTEXT={context}
OUTPUT={output}

GPU=0
SERBAN_IMAGE=ufoym/deepo:theano-py36-cu90
SERBAN_ROOT=/home/cgsdfc/deployment/Models/HRED-VHRED

docker run --rm -it --runtime nvidia \
    --name {name} \
    -v $HOME:$HOME \
    -w $SERBAN_ROOT \
    -e PYTHONPATH=$SERBAN_ROOT \
    -e THEANO_FLAGS=device=cuda$GPU \
    $SERBAN_IMAGE \
    python bin/sample.py \
    $MODEL_PREFIX \
    $CONTEXT \
    $OUTPUT \
    --verbose
