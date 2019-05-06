#!/usr/bin/env bash

IMAGE={adem_image}
PROJECT_PATH={adem_root}
CONTEXTS={contexts}
REFERENCES={references}
RESPONSES={responses}

docker run --runtime nvidia --rm -it \
    -v $HOME:$HOME \
    -w $PROJECT_PATH \
    -e PYTHONPATH=$PROJECT_PATH \
    $IMAGE python entry.py $CONTEXTS $REFERENCES $RESPONSES
