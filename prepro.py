"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

preprocess annotations into LMDB
"""
import argparse
import json
import os
from os.path import exists

from cytoolz import curry
from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm

from data.data import open_lmdb


@curry
def bert_tokenize(tokenizer, text):
    ids = []
    for word in text.strip().split():
        ws = tokenizer.tokenize(word)
        if not ws:
            # some special char
            continue
        ids.extend(tokenizer.convert_tokens_to_ids(ws))
    return ids


def process_cc(json_file, db, tokenizer, missing=None, split="train"):
    id2len, txt2img = {}, {}
    json_file = json.load(json_file)
    for image in tqdm(json_file['images'], desc='processing Conceptual Captions'):
        if image["split"] != split: continue
        img_id = image["imgid"]
        img_fname = image['filename']
        npy_fname = img_fname[:-4] + ".npy"
        for sentence in image["sentences"]:
            if missing and img_fname in missing: continue
            example = sentence
            input_ids = tokenizer(example['raw'])
            id_ =  f"{image['split']}-{img_fname[:-4]}"
            example['id'] = id_
            example['imgid'] = image['imgid']
            example['split'] = image['split']
            example['img_fname'] = npy_fname
            example['raw_fname'] = img_fname
            example['input_ids'] = input_ids

            txt2img[id_] = img_fname
            id2len[id_] = len(input_ids)
            db[id_] = example
    return id2len, txt2img


def process_nlvr2(jsonl, db, tokenizer, missing=None):
    id2len = {}
    txt2img = {}  # not sure if useful
    for line in tqdm(jsonl, desc='processing NLVR2'):
        example = json.loads(line)
        id_ = example['identifier']
        img_id = '-'.join(id_.split('-')[:-1])
        img_fname = (f'nlvr2_{img_id}-img0.npz', f'nlvr2_{img_id}-img1.npz')
        if missing and (img_fname[0] in missing or img_fname[1] in missing):
            continue
        input_ids = tokenizer(example['sentence'])
        if 'label' in example:
            target = 1 if example['label'] == 'True' else 0
        else:
            target = None
        txt2img[id_] = img_fname
        id2len[id_] = len(input_ids)
        example['input_ids'] = input_ids
        example['img_fname'] = img_fname
        example['target'] = target
        db[id_] = example
    return id2len, txt2img


def main(opts):
    if not exists(opts.output):
        os.makedirs(opts.output)
    else:
        raise ValueError('Found existing DB. Please explicitly remove '
                         'for re-processing')
    meta = vars(opts)
    meta['tokenizer'] = opts.toker
    toker = BertTokenizer.from_pretrained(
        opts.toker, do_lower_case='uncased' in opts.toker)
    tokenizer = bert_tokenize(toker)
    meta['UNK'] = toker.convert_tokens_to_ids(['[UNK]'])[0]
    meta['CLS'] = toker.convert_tokens_to_ids(['[CLS]'])[0]
    meta['SEP'] = toker.convert_tokens_to_ids(['[SEP]'])[0]
    meta['MASK'] = toker.convert_tokens_to_ids(['[MASK]'])[0]
    meta['v_range'] = (toker.convert_tokens_to_ids('!')[0],
                       len(toker.vocab))
    with open(f'{opts.output}/meta.json', 'w') as f:
        json.dump(vars(opts), f, indent=4)

    open_db = curry(open_lmdb, opts.output, readonly=False)
    with open_db() as db:
        with open(opts.annotation) as ann:
            if opts.missing_imgs is not None:
                missing_imgs = set(json.load(open(opts.missing_imgs)))
            else:
                missing_imgs = None
            id2lens, txt2img = process_cc(ann, db, tokenizer, missing_imgs, opts.split)

    with open(f'{opts.output}/id2len.json', 'w') as f:
        json.dump(id2lens, f)
    with open(f'{opts.output}/txt2img.json', 'w') as f:
        json.dump(txt2img, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', required=True,
                        help='annotation JSON')
    parser.add_argument('--missing_imgs',
                        help='some training image features are corrupted')
    parser.add_argument('--output', required=True,
                        help='output dir of DB')
    parser.add_argument('--toker', default='bert-base-cased',
                        help='which BERT tokenizer to used')
    parser.add_argument('--split', default='train')
    args = parser.parse_args()
    main(args)
