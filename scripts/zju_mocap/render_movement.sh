#!/usr/bin/env bash

SUBJECT=$1
if [ -z "${SUBJECT}" ]
then
    SUBJECT=387
fi

python run.py \
    --type movement \
    --cfg ./configs/occnerf/zju_mocap/${SUBJECT}/occnerf.yaml \
    load_net latest
