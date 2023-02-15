#!/usr/bin/env bash

set -x

CFG=$1
CHECKPOINT=$2
GPUS=$3
PY_ARGS=${@:4}

# set work_dir according to config path and pretrained model to distinguish different models
WORK_DIR="$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/$(echo $PRETRAIN | rev | cut -d/ -f 1 | rev)"
SAVE_DIR = "/images_temp"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
mim test mmdet \
    $CFG \
    --checkpoint $CHECKPOINT \
    --launcher pytorch \
    --show-dir  "/home/ls/mmselfsupv1/work_dirs/benchmarks/mmdetection/voc0712/faster_rcnn_swin_fpn_voc0712ls_iter_grid_num/images_temp" \
    -G $GPUS \
    --work-dir $WORK_DIR \
    $PY_ARGS

