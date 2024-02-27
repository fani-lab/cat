"""Simple dataset loader for the 2014, 2015 semeval datasets."""
from sklearn.preprocessing import LabelEncoder
from functools import partial


def loader(instance_path,
           label_path,
           label_multi_path,
           subset_labels_path,
           split_labels=False,
           mapping=None):
    # subset_labels = set(subset_labels)

    multi_labels = []
    with open(label_multi_path, 'r') as file:
        for line in file:
            current_array = eval(line.strip())
            multi_labels.append(current_array)

    # multi_labels = open(label_multi_path)
    # multi_labels = [x for x in multi_labels]

    labels = open(label_path)
    labels = [x.strip().lower().split() for x in labels]

    # subset_labels = open(subset_labels_path)
    subset_labels = []
    with open(subset_labels_path, 'r', encoding='utf-8') as file:
        for line in file:
            subset_labels.append(line.strip())
    subset_labels = set([x.strip().lower() for x in subset_labels])
    # print(subset_labels)

    # for test - LADy
    # # subset_labels = {'staff', 'duck', 'confit', 'restaurant'}
    # subset_labels = {'wine', 'place', 'food'}

    instances = []
    for line in open(instance_path, encoding='utf-8'):
        instances.append(line.strip().lower().split())

    if split_labels:
        labels = [[l.split("#")[0] for l in x] for x in labels]

    instances, gold = zip(*[(x, y[0]) for x, y in zip(instances, labels)
                            if len(y) == 1])
                            # and y[0] in subset_labels])

    if mapping is not None:
        gold = [mapping.get(x, x) for x in gold]

    le = LabelEncoder()
    y = le.fit_transform(gold)
    label_set = le.classes_.tolist()

    return instances, y, label_set, subset_labels, gold, multi_labels


# rest_15_test = partial(loader,
#                        instance_path="data/restaurant_test_2015_tok.txt",
#                        label_path="data/labels_restaurant_test_2015.txt",
#                        subset_labels={"ambience",
#                                       "service",
#                                       "food"},
#                        split_labels=True)


def test(f, dataset):
    for h in range(0, 101, 10):
        data_test = partial(loader,
                            instance_path=f"../data/{dataset}/test/{h}/test.txt",
                            label_path=f"../data/{dataset}/test/{h}/test_label.txt",
                            label_multi_path=f"../data/{dataset}/test/{h}/test_label_multi.txt",
                            subset_labels_path=f"../data/{dataset}/train/{f}/train_label.txt",
                            split_labels=True)

        yield data_test()