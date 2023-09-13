import datasets
from datasets import load_dataset
import random
import numpy as np
import csv
import sys
import os

datasets.logging.set_verbosity(datasets.logging.ERROR)

task_to_keys = {
    'clinc150': ("text", None),
    'bank': ("text", None),
    "stackoverflow": ("text", None)
}


def load(task_name, tokenizer, shot=0, max_seq_length=256, is_id=False, dir=None, known_cls_ratio=None):
    sentence1_key, sentence2_key = task_to_keys[task_name]
    print("Loading {}".format(task_name))

    if task_name == 'clinc150':
        datasets, num_labels = load_clinc(is_id, shot=shot, dir=dir, known_cls_ratio=known_cls_ratio)
    elif task_name == 'ROSTD':
        datasets, num_labels = load_clinc(is_id, shot=shot)
    elif task_name == 'bank':
        datasets, num_labels = load_uood(is_id, shot=shot, dir=dir, known_cls_ratio=known_cls_ratio)
    elif task_name == 'stackoverflow':
        datasets, num_labels = load_uood(is_id, shot=shot, dir=dir, known_cls_ratio=known_cls_ratio)
    else:
        print("task is not supported")

    def preprocess_function(examples):
        inputs = (
            (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key] + " " + examples[sentence2_key],)
        )
        result = tokenizer(*inputs, max_length=max_seq_length, truncation=True)
        result["labels"] = examples["label"] if 'label' in examples else 0
        return result

    train_dataset = list(map(preprocess_function, datasets['train'])) if 'train' in datasets and is_id else None
    dev_dataset = list(map(preprocess_function, datasets['validation'])) if 'validation' in datasets and is_id else None
    test_dataset = list(map(preprocess_function, datasets['test'])) if 'test' in datasets else None
    id_dataset = list(map(preprocess_function, datasets['id'])) if 'id' in datasets else None
    return train_dataset, dev_dataset, test_dataset,id_dataset, num_labels


def load_clinc(is_id, shot=None, dir=None, known_cls_ratio=None):
    # label_list = get_labels(dir)
    # label_map = {}
    # for i, label in enumerate(label_list):
    #     label_map[label] = i
    all_label_list_pos = get_labels(dir)
    n_known_cls = round(len(all_label_list_pos) * known_cls_ratio)
    known_label_list = list(
        np.random.choice(np.array(all_label_list_pos), n_known_cls, replace=False))

    ood_labels = list(set(all_label_list_pos) - set(known_label_list))
    label_map = {}
    for i, label in enumerate(known_label_list):
        label_map[label] = i
    label_map["oos"] = n_known_cls

    train_dataset = _create_examples(
        _read_tsv(os.path.join(dir, "train.tsv")), label_map, known_label_list)
    dev_dataset = _create_examples(
        _read_tsv(os.path.join(dir, "dev.tsv")), label_map, known_label_list)
    test_dataset = _create_examples(
        _read_tsv(os.path.join(dir, "test.tsv")), label_map, known_label_list)

    shots = get_shots(shot, train_dataset, "clinc150")
    train_dataset = select_few_shot(shots, train_dataset, "clinc150")
    # dev_dataset = select_few_shot(shot, dev_dataset, "clinc150")
    dev_dataset = select_few_shot(shots, dev_dataset, "clinc150")


    ood_dataset = _get_ood(
        _read_tsv(os.path.join(dir, "test.tsv")), ood_labels + ["oos"], label_map)
    datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset+ood_dataset, "id": test_dataset}
    # datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset, "ood": ood_dataset}
    return datasets, n_known_cls


def load_uood(is_id, shot=None, dir=None, known_cls_ratio=None):
    all_label_list_pos = get_labels(dir)
    n_known_cls = round(len(all_label_list_pos) * known_cls_ratio)
    known_label_list = list(
        np.random.choice(np.array(all_label_list_pos), n_known_cls, replace=False))

    ood_labels = list(set(all_label_list_pos) - set(known_label_list))
    label_map = {}
    for i, label in enumerate(known_label_list):
        label_map[label] = i
    label_map["oos"] = n_known_cls
    train_dataset = _create_examples(
        _read_tsv(os.path.join(dir, "train.tsv")), label_map, known_label_list)
    dev_dataset = _create_examples(
        _read_tsv(os.path.join(dir, "dev.tsv")), label_map, known_label_list)
    test_dataset = _create_examples(
        _read_tsv(os.path.join(dir, "test.tsv")), label_map, known_label_list)

    shots = get_shots(shot, train_dataset, "bank")
    train_dataset = select_few_shot(shots, train_dataset, "bank")
    dev_dataset = select_few_shot(shots, dev_dataset, "bank")

    ood_dataset = _get_ood(
        _read_tsv(os.path.join(dir, "test.tsv")), ood_labels, label_map)
    datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset+ood_dataset, "id": test_dataset}
    return datasets, n_known_cls


def load_ROSTD(is_id, shot=100, data_dir="/data1/liming/ROSTD"):
    label_list = get_labels(data_dir)
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i
    ood_list = ['oos']

    if is_id:
        train_dataset = _create_examples(
            _read_tsv(os.path.join(data_dir, "train.tsv")), label_map, label_list)
        dev_dataset = _create_examples(
            _read_tsv(os.path.join(data_dir, "dev.tsv")), label_map, label_list)
        test_dataset = _create_examples(
            _read_tsv(os.path.join(data_dir, "test.tsv")), label_map, label_list)
        train_dataset = select_few_shot(shot, train_dataset, "clinc150")
        dev_dataset = select_few_shot(shot, dev_dataset, "clinc150")
        datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}
    else:
        test_dataset = _get_ood(
            _read_tsv(os.path.join(data_dir, "test.tsv")), ood_list)
        datasets = {'test': test_dataset}
    return datasets

def get_shots(shot, trainset, task_name):
    # examples = []
    few_examples = []
    sentence1_key, sentence2_key = task_to_keys[task_name]
    from collections import defaultdict
    sorted_examples = defaultdict(list)

    for example in trainset:
        # if example.label in self.known_label_list and np.random.uniform(0, 1) <= args.labeled_ratio:
        #     examples.append(example)
        sorted_examples[example["label"]] = sorted_examples[example["label"]] + [example[sentence1_key]]
    k, v = list(sorted_examples.items())[0]

    return round(len(v) * shot)
    # for k, v in sorted_examples.items():
    #     arr = np.array(v)
    #     shot_n = round(len(arr) * shot)
    #     np.random.shuffle(arr)
    #     for elems in arr[:shot_n]:
    #         few_examples.append({sentence1_key: elems, 'label': k})
    #
    # return few_examples

def select_few_shot(shot, trainset, task_name):
    # examples = []
    few_examples = []
    sentence1_key, sentence2_key = task_to_keys[task_name]
    from collections import defaultdict
    sorted_examples = defaultdict(list)

    for example in trainset:
        # if example.label in self.known_label_list and np.random.uniform(0, 1) <= args.labeled_ratio:
        #     examples.append(example)
        sorted_examples[example["label"]] = sorted_examples[example["label"]] + [example[sentence1_key]]
    for k, v in sorted_examples.items():
        arr = np.array(v)
        # shot_n = round(len(arr) * shot)
        np.random.shuffle(arr)
        for elems in arr[:shot]:
            few_examples.append({sentence1_key: elems, 'label': k})

    return few_examples


def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines


def _create_examples(lines, label_map, know_labels):
    """Creates examples for the training and dev sets."""
    examples = []

    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        if len(line) != 2:
            continue
        # guid = "%s-%s" % (set_type, i)
        text_a = line[0]
        label = line[1]
        if label in know_labels:
            examples.append(
                {'text': text_a, 'label': label_map[label]})
    return examples


def _get_ood(lines, ood_labels, label_map):
    out_examples = []

    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        if len(line) != 2:
            continue
        # guid = "%s-%s" % (set_type, i)
        text_a = line[0]
        label = line[1]
        if label in ood_labels:
            out_examples.append(
                {'text': text_a, 'label': label_map["oos"]})

    return out_examples


def get_labels(data_dir):
    """See base class."""
    import pandas as pd
    test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
    labels = np.unique(np.array(test['label']))

    return labels
