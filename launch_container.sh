# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

TXT_DB=/home/jaredfer/data/txt_db
IMG_DIR=/home/jaredfer/data/img_db
OUTPUT=/home/jaredfer/data/storage
PRETRAIN_DIR=/home/jaredfer/data/pretrained

if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES='ALL'
fi


docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
    --mount src=$(pwd),dst=/src,type=bind \
    --mount src=$OUTPUT,dst=/storage,type=bind \
    --mount src=$PRETRAIN_DIR,dst=/pretrain,type=bind,readonly \
    --mount src=$TXT_DB,dst=/txt,type=bind,readonly \
    --mount src=$IMG_DIR,dst=/img,type=bind,readonly \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -w /src jaredfern/uniter
