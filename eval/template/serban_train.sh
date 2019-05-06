#!/usr/bin/env bash

SAVE_DIR={save_dir}
GPU=0

THEANO_FLAGS=device=cuda$GPU \
python bin/train.py {prototype} \
    --save-dir $SAVE_DIR \
    --prefix {model_prefix} \
    --auto_restart
