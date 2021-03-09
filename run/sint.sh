#!/bin/bash
# Run interactive job to GPU node using Singularity
# Note that users might see the message:
# WARNING: NVIDIA binaries may not be bound with --writable
# And `nvidia-smi` will not be accessible
# Don't worry, one can safely ignore this message.
# https://github.com/sylabs/singularity/issues/2944
DATASETS_DIR="$DATA_DIR/UNITER/datasets"
MODELS_DIR="$DATA_DIR/UNITER/models"
IMG_PATH=$SINGULARITY_IMGS_DIR/uniter_sandbox

SINGULARITY_BIND="$DATA_DIR/UNITER/models/storage:/storage,$UNITER_DIR:/src"
SINGULARITY_BIND="${SINGULARITY_BIND},$DATA_DIR/UNITER/models/pretrained:/pretrained"
SINGULARITY_BIND="${SINGULARITY_BIND},$DATA_DIR/UNITER/datasets/txt_db:/txt"
SINGULARITY_BIND="${SINGULARITY_BIND},$DATA_DIR/UNITER/datasets/img_db:/img"
SINGULARITY_BIND="${SINGULARITY_BIND},$CORPORA_DIR:/corpora,$IMG_PATH/opt:/opt"

srun --mem 64G --time 4-00:00:00 --pty zsh \
    -c "singularity shell --nv -B $SINGULARITY_BIND $IMG_PATH";
