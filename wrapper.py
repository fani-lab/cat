import argparse
import os
import pickle
import json
import numpy as np
from random import random
import re

from cmn.review import Review


def load(reviews, splits):
    print('\nLoading reviews and preprocessing ...')
    print('#' * 50)
    try:
        print('\nLoading reviews files ...')
        with open(f'{reviews}', 'rb') as f:
            reviews = pickle.load(f)
        with open(f'{splits}', 'r') as f:
            splits = json.load(f)
    except (FileNotFoundError, EOFError) as e:
        print(e)
        print('\nLoading existing file failed!')
    print(f'(#reviews: {len(reviews)})')
    return reviews, splits


def preprocess(org_reviews):
    reviews_list = []
    label_list = []
    for r in org_reviews:
        if not len(r.aos[0]):
            continue
        else:
            for aos_instance in r.get_aos():
                for aos in aos_instance[0][0]:
                    reviews_list.append(r.get_txt())
                    label_list.append(aos)
            if r.augs:
                for key, value in r.items():
                    for aos_instance in r[key][1].get_aos():
                        for aos in aos_instance[0][0]:
                            reviews_list.append(r[key][1].get_txt())
                            label_list.append(aos)
    return reviews_list, label_list


# python main.py -ds_name [YOUR_DATASET_NAME] -sgd_lr [YOUR_LEARNING_RATE_FOR_SGD] -win [YOUR_WINDOW_SIZE] -optimizer [YOUR_OPTIMIZER] -rnn_type [LSTM|GRU] -attention_type [bilinear|concat]
def main(args):
    if not os.path.isdir(f'{args.output}'): os.makedirs(f'{args.output}')
    org_reviews, splits = load(args.reviews, args.splits)
    test = np.array(org_reviews)[splits['test']].tolist()

    for h in range(0, 101, 10):

        path = f'{args.output}/{h}/{args.dname}'
        if not os.path.isdir(f'{args.output}/{h}'):
            os.makedirs(f'{args.output}/{h}')

        preprocessed_test, label_list = preprocess(test)

        with open(f'{path}_test_label.txt', 'w') as file:
            for d in label_list:
                file.write(d + '\n')

        hp = h / 100
        test_hidden = []
        for t in range(len(test)):
            if random() < hp:
                test_hidden.append(test[t].hide_aspects())
            else:
                test_hidden.append(test[t])
        preprocessed_test, label_list = preprocess(test_hidden)

        with open(f'{path}_test.txt', 'w') as file:
            for d in preprocessed_test:
                file.write(d + '\n')

    train, label_list = preprocess(np.array(org_reviews)[splits['folds']['0']['train']].tolist(), False)
    path = f'{args.output}/{args.dname}'
    with open(f'{path}_train.txt', 'w') as file:
        for d in train:
            file.write(d + '\n')
    with open(f'{path}_train_label.txt', 'w') as file:
        for d in label_list:
            file.write(d + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CAt Wrapper')
    parser.add_argument('--dname', dest='dname', type=str, default='toy')
    parser.add_argument('--reviews', dest='reviews', type=str,
                        default='data/reviews.pkl',
                        help='raw dataset file path')
    parser.add_argument('--splits', dest='splits', type=str,
                        default='data/splits.json',
                        help='raw dataset file path')
    parser.add_argument('--output', dest='output', type=str,
                        default='data/',
                        help='output path')
    args = parser.parse_args()

    main(args)
