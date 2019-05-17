import pickle
import os
import time
import shutil

import data
from vocab import Vocabulary  # NOQA
from model import VSE
import evaluation

import logging

import argparse

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',
                        help='path to datasets')
    parser.add_argument('--checkpoint', required=True,)
    opt = parser.parse_args()
    print(opt)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    evaluation.evalrank(opt.checkpoint, data_path=opt.data_path, split="test", fold5=True)


if __name__ == '__main__':
    main()
