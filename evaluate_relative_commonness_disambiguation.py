import time
import pickle
from etl_dataset import datasets

with open("./warehouse/spotMapSR.pickle", 'rb') as f:
    d = pickle.load(f)
    spot_map = {}
    for k, v in d.items():
        spot_map[k.lower()] = v
    print('loaded SpotMap')

with open("./warehouse/max_relative_commonness.pickle", 'rb') as f:
    commonness = pickle.load(f)
    print('loaded commonness')


def get_matches(tokens, max_ngram_size=6):
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
    for i in range(max_ngram_size, 0, -1):
        if " ".join(tokens[0:i]) in spot_map:
            return {
                "Wikipedia_ID": commonness[" ".join(tokens[0:i])],
                "anchor_text": " ".join(tokens[0:i]),
                "length": i
            }
    return {
        "Wikipedia_ID": -1,
        "anchor_text": tokens[0],
        "length": 1
    }


def get_total_matches(tokens, max_ngram_size=6):
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
    max_total = 0
    max_match = {
            "Wikipedia_ID": -1,
            "anchor_text": tokens[0],
            "length": 1
        }
    for i in range(max_ngram_size, 0, -1):
        if " ".join(tokens[0:i]) in spot_map:
            if commonness[" ".join(tokens[0:i])]['total'] <10 and max_match['Wikipedia_ID']:
                continue
            if commonness[" ".join(tokens[0:i])]['total'] > max_total:
                max_total = commonness[" ".join(tokens[0:i])]['total']

            max_match = {
                "Wikipedia_ID": commonness[" ".join(tokens[0:i])],
                "anchor_text": " ".join(tokens[0:i]),
                "length": i
            }
    return max_match


def get_fuzzy_matches(tokens, max_ngram_size=6):
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
    matches = {}
    for i in range(max_ngram_size, 0, -1):
        if " ".join(tokens[0:i]) in commonness:
            matches[i] = {
                "Wikipedia_ID": commonness[" ".join(tokens[0:i])],
                "anchor_text": " ".join(tokens[0:i]),
            }
    if not matches:
        matches[max_ngram_size] = {
            "Wikipedia_ID": -1,
            "anchor_text": " ".join(tokens[0:max_ngram_size])
        }
    return matches


def spotter(text, max_ngram_size=10):
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
        m = get_matches(tokens[cursor:cursor + max_ngram_size])
        spots[cursor] = m
        cursor += m['length']
    return spots


def total_spotter(text, max_ngram_size=10):
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
        m = get_total_matches(tokens[cursor:cursor + max_ngram_size])
        spots[cursor] = m
        cursor += m['length']
    return spots


def fuzzy_spotter(text, max_ngram_size=10):
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
        spots[cursor] = get_fuzzy_matches(tokens[cursor:cursor + max_ngram_size])
        cursor += 1
    return spots


def link_and_evaluate_spotted_dataset(dataset, methodology):
    """

    :param dataset:
    :param methodology: One of SR, or SR_norm, SR_min_max_norm
    :return:
    """
    true_prediction = []
    probabilities = []
    for d in dataset:
        anchor = d['mention'].lower()
        if d['mention_position'] in d['spots'] and \
                d['mention_length'] == d['spots'][d['mention_position']]['length'] and \
                d.get('Wikipedia_ID') == commonness.get(anchor, {'id': -1})['id']:
            true_prediction += [1]
            probabilities += [commonness.get(anchor, {methodology: 0.0})[methodology]]
        else:
            true_prediction += [0]
            spoted_mention_position = sorted(list(filter(lambda i: i <= d['mention_position'], d['spots'].keys())))[-1]
            anchor = d['spots'][spoted_mention_position]['anchor_text']
            probabilities += [commonness.get(anchor, {methodology: 0.0})[methodology]]
    return true_prediction, probabilities


def link_and_evaluate_fuzzy_spotted_dataset(dataset, methodology):
    """

    :param dataset:
    :param methodology: One of SR, or SR_norm, SR_min_max_norm
    :return:
    """
    true_prediction = []
    probabilities = []
    for d in dataset:
        anchor = d['mention'].lower()
        if d['mention_position'] in d['spots'] and \
                d['mention_length'] in d['spots'][d['mention_position']] and \
                d['Wikipedia_ID'] == commonness.get(anchor, {'id': -1})['id']:
            true_prediction += [1]
            probabilities += [commonness[anchor][methodology]]
        else:

            true_prediction += [0]
            spoted_mention_position = sorted(list(filter(lambda i: i <= d['mention_position'], d['spots'].keys())))[-1]

            # average probability within fuzzy set:
            prob = 0.0
            for spot in d['spots'][spoted_mention_position].values():
                anchor = spot['anchor_text']
                prob += commonness.get(anchor, {methodology: 0.0})[methodology]
            probabilities += [prob/len(d['spots'][spoted_mention_position])]
    return true_prediction, probabilities


def spot_dataset(dataset, spotter):
    """
    Creates dataset spots
    :param spotter:  Spotter function
    :param dataset:
    :return:
    """
    processed_dataset = []
    for di in dataset:
        text = di['context_left'] + ' ' + di['mention'] + ' ' + di['context_right']
        d = di.copy()
        d['spots'] = spotter(text.lower())
        processed_dataset.append(d)
    return processed_dataset


def pckl(path, data):
    with open(path, 'wb') as fp:
        print('Saving', path)
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


def evaluate_run_time(dataset, method):
    """

    :param method: The method being time evaluated
    :param dataset: The dataset for time evaluation
    :return:
    """
    start = time.time()
    for di in dataset:
        text = (di['context_left'] + ' ' + di['mention'] + ' ' + di['context_right']).lower()
        spots = spotter(text, max_ngram_size=6)
        for s in spots.values():
            outcome = commonness.get(s['anchor_text'], {method: 0.0})[method]
    timing = time.time() - start
    print("--- %s seconds ---" % timing)


def evaluate_fuzzy_run_time(dataset, method):
    """

    :param method: The method being time evaluated
    :param dataset: The dataset for time evaluation
    :return:
    """
    start = time.time()
    for di in dataset:
        text = (di['context_left'] + ' ' + di['mention'] + ' ' + di['context_right']).lower()
        spots = fuzzy_spotter(text, max_ngram_size=6)
        for s in spots.values():
            for i in s.values():
                outcome = commonness.get(i['anchor_text'], {method: 0.0})[method]
    timing = time.time() - start
    print("---- %s seconds ---" % timing)

if __name__ == '__main__':
    for name, dtset in datasets.items():
        print(name, '- relative commonness')
        evaluate_run_time(dtset, 'relative_commonness')
        print('fuzzy')
        evaluate_fuzzy_run_time(dtset, 'relative_commonness')
        spotted_dataset = spot_dataset(dtset, spotter)
        y_true, probs = link_and_evaluate_spotted_dataset(spotted_dataset, 'relative_commonness')
        pckl('results/' + name + '_relative_commonness_y_true', y_true)
        pckl('results/' + name + '_relative_commonness_probs', probs)
        fuzzy_spotted_dataset = spot_dataset(dtset, fuzzy_spotter)
        fuzzy_y_true, fuzzy_probs = link_and_evaluate_fuzzy_spotted_dataset(fuzzy_spotted_dataset, 'relative_commonness')
        pckl('results/' + name + '_fuzzy_relative_commonness_y_true', fuzzy_y_true)
        pckl('results/' + name + '_fuzzy_relative_commonness_probs', fuzzy_probs)
