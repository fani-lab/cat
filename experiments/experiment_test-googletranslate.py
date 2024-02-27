"""Experiment on the test data."""
import json
import os

import numpy as np

# LADy_eval
import pytrec_eval
import pandas as pd

from cat.simple import get_scores, attention, rbf_attention
from cat.dataset import test
from reach import Reach
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict, Counter
from itertools import product


GAMMA = .03
BEST_ATT = {"n_noun": 980}
BEST_RBF = {"n_noun": 200}

if __name__ == "__main__":

    # LADy_eval
    metrics = ['P', 'recall', 'ndcg_cut', 'map_cut', 'success']
    topkstr = '1,5,10,100'
    metrics_set = set()
    for m in metrics:
        metrics_set.add(f'{m}_{topkstr}')
    datasets = []
    for d in ['googletranslate-twitter']:  # 'googletranslate-SemEval-14-L'
        for l in ['en', 'fa', 'zh-CN', 'de', 'ar', 'fr', 'es', 'fa.zh-CN.de.ar.fr.es']: #
            if l == 'en':
                datasets.append(f'{d}')
            else:
                datasets.append(f'{d}-{l}')
    for dataset in datasets:

        output_path = f'../output-googletranslate/{dataset}'
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        mean_list = [pd.DataFrame() for i in range(0, 11)]
        for f in range(5):
            fold_path = f'{dataset}/train/{f}'
            scores = defaultdict(dict)
            r = Reach.load(f'../embeddings/{fold_path}/vecs_w2v.vec',
                           unk_word="<UNK>")
            d = json.load(open(f'../data/{fold_path}/nouns.json'))

            nouns = Counter()
            for k, v in d.items():
                if k.lower() in r.items:
                    nouns[k.lower()] += v

            embedding_paths = [f'../embeddings/{fold_path}/vecs_w2v.vec']
            # bundles = ((rbf_attention, attention), embedding_paths)
            bundles = ((rbf_attention, ), embedding_paths)

            for att, path in product(*bundles):
                r = Reach.load(path, unk_word="<UNK>")

                if att == rbf_attention:
                    candidates, _ = zip(*nouns.most_common(BEST_RBF["n_noun"]))
                else:
                    candidates, _ = zip(*nouns.most_common(BEST_ATT["n_noun"]))

                aspects = [[x] for x in candidates]
                sorted_output = []
                for idx, (instances, y, label_set, subset_labels, gold, multi_labels) in enumerate(test(f, dataset)):
                    # output_path_hidden = f'{output_path}/{idx*10}'
                    # if not os.path.isdir(output_path_hidden):
                    #     os.makedirs(output_path_hidden)
                    # print("label_set", label_set)
                    s = get_scores(instances,
                                   aspects,
                                   r,
                                   subset_labels,
                                   gamma=GAMMA,
                                   remove_oov=False,
                                   attention_func=att)

                    # print("predicted", s)
                    # print("subset_labels", subset_labels)
                    # print("gold", list(gold))
                    output = [[(label, value) for value, label in zip(sublist, subset_labels)] for sublist in s]
                    sorted_output = [sorted(sublist, key=lambda x: x[1], reverse=True) for sublist in output]
                    # print("output", sorted_output)

                    qrel = dict()
                    run = dict()

                    for i, word in enumerate(multi_labels):
                        q_key = 'q{}'.format(i)
                        # qrel[q_key] = {word: 1}
                        qrel[q_key] = {w: 1 for w in word}

                    for i, sublist in enumerate(sorted_output):
                        q_key = 'q{}'.format(i)
                        run[q_key] = {}
                        for j, (word, _) in enumerate(sublist):
                            run[q_key][word] = len(sublist) - j

                    # print("qrel: ", qrel)
                    # print("run: ", run)

                    print(f'pytrec_eval for {metrics_set} for fold {f} with {idx*10} percent hidden aspect in dataset {dataset}...')
                    df = pd.DataFrame.from_dict(pytrec_eval.RelevanceEvaluator(qrel, metrics_set).evaluate(run))
                    df_mean = df.mean(axis=1).to_frame('mean')
                    df_mean.to_csv(f'{output_path}/f{f}.model.ad.pred.{idx/10}.eval.mean.csv')
                    mean_list[idx] = pd.concat([mean_list[idx], df_mean], axis=1)
        for i in range(0, 11):
            # output_path_hidden = f'{output_path}/{i*10}'
            mean_list[i].mean(axis=1).to_frame('mean').to_csv(f'{output_path}/model.ad.pred.eval.mean.{i/10}.csv')
            # y_pred = s.argmax(1)
    #         f1_score = precision_recall_fscore_support(y, y_pred)
    #         f1_macro = precision_recall_fscore_support(y,
    #                                                    y_pred,
    #                                                    average="weighted")
    #         scores[(att, path)][idx] = (f1_score, f1_macro)
    #
    # att_score = {k: v for k, v in scores.items() if k[0] == attention}
    # att_per_class = [[z[x][0][:-1] for x in range(3)]
    #                  for z in att_score.values()]
    # att_per_class = np.stack(att_per_class).mean(0)
    # att_macro = np.mean([v[2][1][:-1] for v in att_score.values()], 0)
    #
    # rbf_score = {k: v for k, v in scores.items() if k[0] == rbf_attention}
    # rbf_per_class = [[z[x][0][:-1] for x in range(3)]
    #                  for z in rbf_score.values()]
    # rbf_per_class = np.stack(rbf_per_class).mean(0)
    # rbf_macro = np.mean([v[2][1][:-1] for v in rbf_score.values()], 0)
