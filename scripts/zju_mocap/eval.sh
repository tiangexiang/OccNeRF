#!/usr/bin/env bash

SUBJECT=$1
if [ -z "${SUBJECT}" ]
then
    SUBJECT=387
fi

python eval.py \ 
    --cfg ./configs/occnerf/zju_mocap/${SUBJECT}/occnerf.yaml