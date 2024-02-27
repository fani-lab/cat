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
    print('_' * 50)
    try:
        with open(f'{reviews}', 'rb') as f:
            reviews = pickle.load(f)
        with open(f'{splits}', 'r') as f:
            splits = json.load(f)
    except (FileNotFoundError, EOFError) as e:
        print(e)
        print('\nLoading existing file failed!')
    print(f'(#reviews: {len(reviews)})')
    return reviews, splits


def get_aos_augmented(review):
    r = []
    if not review.aos: return r
    for i, aos in enumerate(review.aos): r.append([([review.sentences[i][j] for j in a], [review.sentences[i][j] for j in o], s) for (a, o, s) in aos])
    return r


def preprocess(org_reviews, status, lang):
    reviews_list = []
    label_list = []
    for r in org_reviews:
        if not len(r.aos[0]):
            continue
        else:

            if status == 'test':
                reviews_list.append(r.get_txt())
                label_list.append(r.get_aos()[0][0][0][0])

            elif status == 'multi-test':  # test should not be duplicated in case of having more than one aspect
                reviews_list.append(r.get_txt())
                label_per_review = []
                for aos in r.get_aos()[0][0][0]:
                    label_per_review.append(aos)
                label_list.append(label_per_review)

            else:  # train should be duplicated in case of having more than one aspect
                for aos in r.get_aos()[0][0][0]:
                    text = r.get_txt()
                    label = aos
                    reviews_list.append(text)
                    label_list.append(label)

            '''
            if len(r.get_aos()[0][0][0]) == 1:
                text = r.get_txt()
                label = r.get_aos()[0][0][0][0]
                reviews_list.append(text)
                label_list.append(label)
            '''

            if r.augs and status == 'train':  # data for train can be augmented

                # if lang == 'pes_Arab.zho_Hans.deu_Latn.arb_Arab.fra_Latn.spa_Latn':
                if lang == 'fa.zh-CN.de.ar.fr.es':
                    for key, value in r.augs.items():
                        for aos_instance in get_aos_augmented(r.augs[key][1])[0][0][0]:
                            text = r.augs[key][1].get_txt()
                            label = aos_instance
                            reviews_list.append(text)
                            label_list.append(label)
                else:
                    # for l in lang.split('.'):
                    #     for aos_instance in get_aos_augmented(r.augs[l][1])[0][0][0]:
                    #         text = r.augs[l][1].get_txt()
                    #         label = aos_instance
                    #         reviews_list.append(text)
                    #         label_list.append(label)
                    for aos_instance in get_aos_augmented(r.augs[lang][1])[0][0][0]:
                        text = r.augs[lang][1].get_txt()
                        label = aos_instance
                        reviews_list.append(text)
                        label_list.append(label)
                '''
                # for key, value in r.augs.items():
                    # if len(get_aos_augmented(r.augs[key][1])) == 0:
                    #     text = r.augs[key][1].get_txt()
                    #     reviews_list.append(text)
                    #     continue
                    # for aos_instance in r.augs[key][1].get_aos()[0]:
                if lang == 'pes_Arab.zho_Hans.deu_Latn.arb_Arab.fra_Latn.spa_Latn':
                    for key, value in r.augs.items():
                        if len(get_aos_augmented(r.augs[key][1])[0][0][0]) == 1:
                            text = r.augs[key][1].get_txt()
                            label = get_aos_augmented(r.augs[key][1])[0][0][0][0]
                            reviews_list.append(text)
                            # label_list.append(label)
                elif len(get_aos_augmented(r.augs[lang][1])[0][0][0]) == 1:
                    text = r.augs[lang][1].get_txt()
                    label = get_aos_augmented(r.augs[lang][1])[0][0][0][0]
                    reviews_list.append(text)
                    # label_list.append(label)
                '''

    return reviews_list, label_list


def main(args):
    output_path = f'{args.output}/{args.dname}/'
    if not os.path.isdir(output_path): os.makedirs(output_path)
    org_reviews, splits = load(args.reviews, args.splits)

    for f in range(5):
        path = f'{output_path}train/{f}/'
        if not os.path.isdir(path): os.makedirs(path)
        train, label_list = preprocess(np.array(org_reviews)[splits['folds'][str(f)]['train']].tolist(), 'train', args.lang)

        with open(f'{path}train.txt', 'w', encoding='utf-8') as file:
            for d in train:
                file.write(d + '\n')
        with open(f'{path}train_label.txt', 'w', encoding='utf-8') as file:
            for d in label_list:
                file.write(d + '\n')

    test = np.array(org_reviews)[splits['test']].tolist()

    for h in range(0, 101, 10):

        path = f'{output_path}/test/{h}/'
        if not os.path.isdir(path):
            os.makedirs(path)

        preprocessed_test, label_list = preprocess(test, 'test', args.lang)

        with open(f'{path}test_label.txt', 'w', encoding='utf-8') as file:
            for d in label_list:
                file.write(d + '\n')

        _, labels_list = preprocess(test, 'multi-test', args.lang)
        with open(f'{path}test_label_multi.txt', 'w', encoding='utf-8') as file:
            for d in labels_list:
                file.write(str(d) + '\n')

        hp = h / 100
        test_hidden = []
        for t in range(len(test)):
            if random() < hp:
                test_hidden.append(test[t].hide_aspects())
            else:
                test_hidden.append(test[t])
        preprocessed_test, label_list = preprocess(test_hidden, 'test', args.lang)

        with open(f'{path}test.txt', 'w', encoding='utf-8') as file:
            for d in preprocessed_test:
                file.write(d + '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CAt Wrapper')
    parser.add_argument('--dname', dest='dname', type=str, default='SemEval-14-R')
    parser.add_argument('--reviews', dest='reviews', type=str,
                        default='data/2015SB12/reviews.pes_Arab.pkl',
                        help='raw dataset file path')
    parser.add_argument('--splits', dest='splits', type=str,
                        default='data/2015SB12/splits.json',
                        help='raw dataset file path')
    parser.add_argument('--output', dest='output', type=str,
                        default='data',
                        help='output path')
    parser.add_argument('--lang', dest='lang', type=str,
                        default='eng',
                        help='language')
    args = parser.parse_args()

    # 'SemEval14L','SemEval14R', '2015SB12', '2016SB5'
    # 'output-twitter-modified'
    # 'googletranslate-2015SB12','googletranslate-2016SB5','googletranslate-SemEval-14-L'
    # for dataset in ['googletranslate-2015SB12','googletranslate-2016SB5','googletranslate-SemEval-14-L','googletranslate-SemEval-14-R', 'googletranslate-twitter']:
    for dataset in ['lowresource-2015', 'lowresource-2016', 'lowresource-2014l', 'lowresource-2014r']:
        args.splits = f'data/{dataset}/splits.json'
        # for lang in []:'eng', 'pes_Arab', 'zho_Hans', 'deu_Latn', 'arb_Arab', 'fra_Latn', 'spa_Latn',
        #         #              'pes_Arab.zho_Hans.deu_Latn.arb_Arab.fra_Latn.spa_Latn'
        for lang in ['lao_Laoo', 'san_Deva']:
            # if lang == 'en':
            if lang == 'eng':
                args.lang = lang
                args.dname = f'{dataset}'
                args.reviews = f'data/{dataset}/reviews.pkl'
            else:
                args.lang = lang
                args.dname = f'{dataset}-{lang}'
                args.reviews = f'data/{dataset}/reviews.{lang}.pkl'
            print(args)
            main(args)
