#!/usr/bin/env bash

SUBJECT=$1
if [ -z "${SUBJECT}" ]
then
    SUBJECT=387
fi

# render T-pose
python run.py \
    --type tpose \
    --cfg ./configs/occnerf/zju_mocap/${SUBJECT}/occnerf.yaml \
    load_net latest
