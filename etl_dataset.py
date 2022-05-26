import json
import os

CDIR = os.path.dirname(os.path.abspath(__file__))


def load_aida_jsonl(fpath):
    """
    Loads and returns jsonl in list format.
    :param fpath:
    :return:
    """
    with open(fpath, 'r') as f:
        return [json.loads(ln) for ln in f.readlines()]


def restructure_dataset(dataset):
    """
    Re-formats the dataset to the following format:
    {
        "context_left":"...",
        "mention":"...",
        "context_right":"...",
        "query_id":"...",
        "label_id":...,
        "Wikipedia_ID":...,
        "Wikipedia_URL":"...",
        "Wikipedia_title":"...",
        "mention_position":...,
        "mention_length":...,
    }
    :param dataset:
    :return:
    """
    restructured_dataset = []
    for d in dataset:
        d['mention_position'] = len(d['context_left'].split())
        d['mention_length'] = len(d['mention'].split())
        restructured_dataset.append(d)
    return restructured_dataset


testa = restructure_dataset(
    load_aida_jsonl(os.path.join(CDIR, 'blink', 'data', 'BLINK_benchmark', 'AIDA-YAGO2_testa.jsonl')))
testb = restructure_dataset(
    load_aida_jsonl(os.path.join(CDIR, 'blink', 'data', 'BLINK_benchmark', 'AIDA-YAGO2_testb.jsonl')))
print('loaded...')

