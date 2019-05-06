#!/usr/bin/env bash

MODEL_PREFIX={model_prefix}
CONTEXT={context}
OUTPUT={output}

GPU=0

THEANO_FLAGS=device=cuda$GPU \
python bin/sample.py \
    $MODEL_PREFIX \
    $CONTEXT \
    $OUTPUT \
    --verbose
