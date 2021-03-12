"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

convert image npz/npy to LMDB
"""
import argparse
import glob
import io
import json
import multiprocessing as mp
import os
from os.path import basename, exists

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
from cytoolz import curry
from PIL import Image
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)
from tqdm import tqdm

msgpack_numpy.patch()


def _compute_nbb(img_dump, conf_th, max_bb, min_bb, num_bb):
    num_bb = max(min_bb, (img_dump > conf_th).sum())
    num_bb = min(max_bb, num_bb)
    return int(num_bb)


def _normalize_bb(bbox, width, height, format='xywh'):
    if format == 'xyxy':
        bbox_x = np.array([bbox[:, 0], bbox[:, 2], bbox[:, 2] - bbox[:, 0]]) / width
        bbox_y = np.array([bbox[:, 1], bbox[:, 3], bbox[:, 3] - bbox[:, 1]]) / height
    if format == 'xywh':
        bbox_x = np.array([bbox[:, 0], bbox[:, 0] + bbox[:, 2], bbox[:, 2]]) / width
        bbox_y = np.array([bbox[:, 1], bbox[:, 1] + bbox[:, 3], bbox[:, 3]]) / height
    return np.vstack((bbox_x[0], bbox_y[0],
                      bbox_x[1], bbox_y[1],
                      bbox_x[2], bbox_y[2])).T


@curry
def load_npy(conf_th, max_bb, min_bb, num_bb, fname, keep_all=False):
    try:
        img_dump = np.load(fname, allow_pickle=True).item()
        if keep_all:
            nbb = None
        else:
            nbb = _compute_nbb(img_dump['cls_prob'], conf_th, max_bb, min_bb, num_bb)

        dump = {}
        dump['norm_bb'] = _normalize_bb(
            img_dump['bbox'], img_dump['image_width'], img_dump['image_height'])

        for key, arr in img_dump.items():
            if type(arr) != np.ndarray:
                continue
            if arr.dtype == np.float32:
                arr = arr.astype(np.float16)
            if arr.ndim == 2:
                dump[key] = arr[:nbb, :]
            elif arr.ndim == 1:
                dump[key] = arr[:nbb]
            else:
                raise ValueError('wrong ndim')
    except Exception as e:
        # corrupted file
        print(f'corrupted file {fname}', e)
        dump = {}
        nbb = 0

    name = basename(fname)
    return name, dump, nbb


@curry
def load_clip_npy(bboxes, feature_type, fname):
    try:
        dump = {}
        if feature_type == 'patches':
            dump['features'] = np.load(fname, allow_pickle=True)[1:]
            dump['norm_bb'] = bboxes
            nbb = 49
        elif feature_type == 'cls':
            dump['features'] = np.load(fname, allow_pickle=True)[None, 0, :]
            dump['norm_bb'] = bboxes
            nbb = 1
        dump['conf'] = np.ones((len(dump['features']), 1))
    except Exception as e:
        print(f'corrupted file {fname}', e)
        dump = {}

    name = basename(fname)
    return name, dump, nbb


@curry
def load_image_tensor(bboxes, input_resolution, fname):
    preprocess = Compose([
        Resize(input_resolution, interpolation=Image.BICUBIC),
        CenterCrop(input_resolution),
        ToTensor()
    ])

    dump = {}
    try:
        dump['image_tensors'] = preprocess(Image.open(fname).convert("RGB"))
        dump['norm_bb'] = bboxes
        dump['conf'] = np.ones((len(dump['features']), 1))
        nbb = len(bboxes)
    except Exception as e:
        print(f'corrupted file {fname}', e)
        dump = {}
    fname = basename(fname)
    return fname, dump, nbb


@curry
def load_npz(conf_th, max_bb, min_bb, num_bb, fname, keep_all=False):
    try:
        img_dump = np.load(fname, allow_pickle=True)
        if keep_all:
            nbb = None
        else:
            nbb = _compute_nbb(img_dump['conf'], conf_th, max_bb, min_bb, num_bb)
        dump = {}
        for key, arr in img_dump.items():
            if arr.dtype == np.float32:
                arr = arr.astype(np.float16)
            if arr.ndim == 2:
                dump[key] = arr[:nbb, :]
            elif arr.ndim == 1:
                dump[key] = arr[:nbb]
            else:
                raise ValueError('wrong ndim')
    except Exception as e:
        # corrupted file
        print(f'corrupted file {fname}', e)
        dump = {}
        nbb = 0

    name = basename(fname)
    return name, dump, nbb


def dumps_npz(dump, compress=False):
    with io.BytesIO() as writer:
        if compress:
            np.savez_compressed(writer, **dump, allow_pickle=True)
        else:
            np.savez(writer, **dump, allow_pickle=True)
        return writer.getvalue()


def dumps_msgpack(dump):
    return msgpack.dumps(dump, use_bin_type=True)


def main(opts):
    if opts.img_dir[-1] == '/':
        opts.img_dir = opts.img_dir[:-1]
    split = basename(opts.img_dir)
    if opts.keep_all:
        db_name = 'all'
    else:
        if opts.conf_th == -1:
            db_name = f'feat_numbb{opts.num_bb}'
        else:
            db_name = (f'feat_th{opts.conf_th}_max{opts.max_bb}'
                       f'_min{opts.min_bb}')
    if opts.compress:
        db_name += '_compressed'
    if not exists(f'{opts.output}/{split}'):
        os.makedirs(f'{opts.output}/{split}')
    env = lmdb.open(f'{opts.output}/{split}/{db_name}', map_size=1024**4)
    txn = env.begin(write=True)
    files = glob.glob(f'{opts.img_dir}/*.npy')
    if opts.feature_format == 'npy_patches':
        info = np.load(opts.info_file, allow_pickle=True).item()
        if opts.feature_type == "cls":
            bboxes = np.array([[0, 0, 1, 1, 1, 1]])
        else:
            bboxes = _normalize_bb(
                info['bbox'], info['image_width'], info['image_height'], format='xyxy')
        load = load_clip_npy(bboxes, opts.feature_type)
    elif opts.feature_format == 'raw_image':
        info = np.load(opts.info_file, allow_pickle=True).item()
        bboxes = np.vstack((
            np.array([0, 0, 1, 1, 1, 1]),
            _normalize_bb(info['bbox'], info['image_width'], info['image_height'], format='xyxy')
        ))
        load = load_image_tensor(bboxes, opts.input_resolution)
    elif opts.feature_format == 'npy':
        load = load_npy(opts.conf_th, opts.max_bb, opts.min_bb, opts.num_bb,
                        keep_all=opts.keep_all)
    elif opts.feature_format == 'npz':
        load = load_npz(opts.conf_th, opts.max_bb, opts.min_bb, opts.num_bb,
                        keep_all=opts.keep_all)
    name2nbb = {}
    with mp.Pool(opts.nproc) as pool, tqdm(total=len(files)) as pbar:
        for i, (fname, features, nbb) in enumerate(
                pool.imap_unordered(load, files, chunksize=128)):
            if not features:
                continue  # corrupted feature
            if opts.compress:
                dump = dumps_npz(features, compress=True)
            else:
                dump = dumps_msgpack(features)
            txn.put(key=fname.encode('utf-8'), value=dump)
            if i % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)
            name2nbb[fname] = nbb
            pbar.update(1)
        txn.put(key=b'__keys__',
                value=json.dumps(list(name2nbb.keys())).encode('utf-8'))
        txn.commit()
        env.close()
    if opts.conf_th != -1 and not opts.keep_all:
        with open(f'{opts.output}/{split}/'
                  f'nbb_th{opts.conf_th}_'
                  f'max{opts.max_bb}_min{opts.min_bb}.json', 'w') as f:
            json.dump(name2nbb, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default=None, type=str,
                        help="The input images.")
    parser.add_argument("--output", default=None, type=str,
                        help="output lmdb")
    parser.add_argument('--nproc', type=int, default=8,
                        help='number of cores used')
    parser.add_argument('--compress', action='store_true',
                        help='compress the tensors')
    parser.add_argument('--keep_all', action='store_true',
                        help='keep all features, overrides all following args')
    parser.add_argument('--feature_format', choices=['npy', 'npz', 'npy_patches', 'raw_image'],
                        help='format of the feature file/associated metadata')
    parser.add_argument('--feature_type', choices=['patches', 'cls'])
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=100,
                        help='number of bounding boxes (fixed)')
    parser.add_argument('--input_resolution', type=int, default=224)
    parser.add_argument('--info_file', type=str, default=None,
                        help='metadata file for all images. used for patches')
    args = parser.parse_args()
    main(args)
