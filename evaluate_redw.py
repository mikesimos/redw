# This file is part of fast-wikification.
#
# Fast-wikification is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.

# Fast-wikification is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.

# You should have received a copy of the GNU General Public License along with
# fast-wikification. If not, see <https://www.gnu.org/licenses/>.

# -*- coding: utf-8 -*-
import pickle
from etl_dataset import datasets

with open("./warehouse/spotMapSR.pickle", 'rb') as f:
    spot_map = pickle.load(f)


def get_matches(tokens, max_ngram_size=10):
    """
    Returns continuous matches with SpotMap for the specific token sequence, starting at the first token.
    Given the tokens *t0,t1,t2,t3,t4,t5,t6,t7,t8,t9* and a max_ngram_size of *5* all candidate matches are
    t0,t1,t2,t3,t4
    t0,t1,t2,t3
    t0,t1,t2
    t0,t1
    t0
    :param tokens:
    :param max_ngram_size:
    :return:
    """
    for i in range(max_ngram_size, -1, -1):
        if " ".join(tokens[0:i]) in spot_map:
            return {
                "anchor_id": spot_map[" ".join(tokens[0:i])],
                "anchor_text": " ".join(tokens[0:i]),
                "length": i
            }
    return {
        "anchor_id": -1,
        "anchor_text": tokens[0],
        "length": 1
    }


def redw_spotter(text, max_ngram_size=10):
    """
    Spots text fragments according to RedW methodology
    :param text:
    :param max_ngram_size:
    :return:
    """
    tokens = text.split()
    l = len(tokens)
    cursor = 0
    spots = {}
    while cursor < l:
        m = get_matches(tokens[cursor:min(cursor + max_ngram_size, l)])
        spots[cursor] = m
        cursor += m['length']
    return spots


def redw_link_and_evaluate_spotted_dataset(dataset, methodology):
    """

    :param dataset:
    :param methodology: One of SR, or SR_norm, SR_min_max_norm
    :return:
    """
    true_prediction = []
    probabilities = []
    for d in dataset:
        if d['mention_position'] in d['spots'] and \
                d['mention_length'] == d['spots'][d['mention_position']]['length'] and \
                d['Wikipedia_ID'] == int(spot_map.get(d['mention'], {'id': -1})['id']):
            true_prediction += [1]
            probabilities += [spot_map.get(d['mention'], {methodology: 0.0})[methodology]]
        else:

            true_prediction += [0]
            spoted_mention_position = sorted(list(filter(lambda i: i <= d['mention_position'], d['spots'].keys())))[-1]
            anchor = d['spots'][spoted_mention_position]['anchor_text']
            probabilities += [spot_map.get(anchor, {methodology: 0.0})[methodology]]
    return true_prediction, probabilities


def redw_spot_dataset(dataset, spotter):
    """
    Creates dataset spots
    :param spotter:  Spotter function
    :param dataset:
    :return:
    """
    spotted_dataset = []
    for d in dataset:
        text = d['context_left'] + ' ' + d['mention'] + ' ' + d['context_right']
        d['spots'] = spotter(text)
        spotted_dataset.append(d)
    return spotted_dataset


def pckl(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def evaluate_run_time(dtset, mthd):
    """

    :param mthd: Method for run-time evaluation
    :param dtset: Dataset for run-time evaluation
    :return:
    """
    import time
    start = time.time()
    for d in dtset:
        text = (d['context_left'] + ' ' + d['mention'] + ' ' + d['context_right']).lower()
        spots = redw_spotter(text)
        for s in spots.values():
            o = spot_map.get(s['anchor_text'], {mthd: 0.0})[mthd]
    timing = time.time() - start
    print ("--- -%s seconds --" % timing)


if __name__ == '__main__':
    for name, dataset in datasets.items():
        spotted_dataset = redw_spot_dataset(dataset, redw_spotter)
        for method in ['SR_norm', 'SR_min_max_norm']:
            y_true, probs = redw_link_and_evaluate_spotted_dataset(spotted_dataset, method)
            pckl('results/' + name + '_redw_' + method + '_y_true', y_true)
            pckl('results/' + name + '_redw_' + method + '_probs', probs)
            print(name, '-', method)
            evaluate_run_time(dataset, method)
