#!/bin/bash
#SBATCH --job-name uniter_base
#SBATCH --output log/pretrain_uniter_base.log
#SBATCH --time 8-00:00:00
#SBATCH --mem 32G
#SBATCH --gres=gpu:4

CONFIG_FILE="$1"

IMG_PATH="${SINGULARITY_IMGS_DIR}/uniter_sandbox"
HOME_DIR=$UNITER_DIR

BINDINGS="$DATA_DIR/UNITER/models/storage/:/storage"
BINDINGS="${BINDINGS},$DATA_DIR/UNITER/models/pretrained/:/pretrained"
BINDINGS="${BINDINGS},$DATA_DIR/UNITER/datasets/txt_db/:/txt"
BINDINGS="${BINDINGS},$DATA_DIR/UNITER/datasets/img_db/:/img"
BINDINGS="${BINDINGS},$CORPORA_DIR/:/corpora,$IMG_PATH/opt/:/opt"

echo $CONFIG_FILE
singularity exec --writable --nv -B $BINDINGS -H $UNITER_DIR $IMG_PATH \
  horovodrun -np 4 python test.py;
