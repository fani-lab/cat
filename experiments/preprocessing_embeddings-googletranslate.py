"""Creating fragments takes a long time so we treat it as a
pre-processing step."""
import logging
import os

from gensim.models import Word2Vec
from cat.fragments import create_noun_counts
from cat.utils import conll2text

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    datasets = []
    for d in ['googletranslate-twitter']:
        for l in ['en', 'fa', 'zh-CN', 'de', 'ar', 'fr', 'es', 'fa.zh-CN.de.ar.fr.es']:
            if l == 'en':
                datasets.append(f'{d}')
            else:
                datasets.append(f'{d}-{l}')
    for dataset in datasets:
        for f in range(5):
            fold_path = f'{dataset}/train/{f}'
            paths = [f'../data/{fold_path}/input.conllu']
            create_noun_counts(paths, f'../data/{fold_path}/nouns.json')
            conll2text(paths, f'../data/{fold_path}/all_txt.txt')
            corpus = [x.lower().strip().split()
                      for x in open(f'../data/{fold_path}/all_txt.txt', encoding='utf-8')]

            f = Word2Vec(corpus,
                         sg=0,
                         negative=5,
                         window=10,
                         vector_size=200,
                         min_count=2,
                         epochs=40,
                         workers=10)
            embedding_path = f"../embeddings/{fold_path}"
            if not os.path.isdir(embedding_path):
                os.makedirs(embedding_path)
            f.wv.save_word2vec_format(f'{embedding_path}/vecs_w2v.vec')
