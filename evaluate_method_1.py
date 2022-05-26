import json
import os
import pickle
from etl_dataset import testa, testb
#
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
        "Wikipedia_ID": None,
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
            "Wikipedia_ID": None,
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
            "Wikipedia_ID": None,
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
        m = get_fuzzy_matches(tokens[cursor:cursor + max_ngram_size])
        spots[cursor] = m
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
                int(d['Wikipedia_ID']) == commonness.get(anchor, {'id': None})['id']:
            true_prediction += [1]
            probabilities += [commonness[anchor][methodology]]
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
                int(d['Wikipedia_ID']) == commonness.get(anchor, {'id': None})['id']:
            true_prediction += [1]
            probabilities += [commonness[anchor][methodology]]
        else:

            true_prediction += [0]
            spoted_mention_position = sorted(list(filter(lambda i: i <= d['mention_position'], d['spots'].keys())))[-1]

            # average probability within fuzzy set:
            prob = 0.0
            for spot in d['spots'][spoted_mention_position].values():
                anchor = spot['anchor_text']
                # if commonness.get(anchor, {methodology: 0.0})[methodology]>prob: #<
                #     prob=commonness.get(anchor, {methodology: 0.0})[methodology] #<
                prob += commonness.get(anchor, {methodology: 0.0})[methodology]
            probabilities += [prob/len(d['spots'][spoted_mention_position])] #<  [prob]
    return true_prediction, probabilities

def evaluate_spotter(spotted_dataset):
    """
    Performs an evaluation for the spotter
    :param spotted_dataset:
    :return:
    """
    hits = 0
    misses = 0
    ms = []
    i = 0
    for d in spotted_dataset:
        if i>0:
            i-=1
            print (d)
        if d['mention_position'] in d['spots'] and \
                d['spots'][d['mention_position']]['length'] == d['mention_length']:
            if d['spots'][d['mention_position']].get('Wikipedia_ID'):
                if int(d['spots'][d['mention_position']].get('Wikipedia_ID')['id'])==int(d['Wikipedia_ID']):
                    hits += 1
                else:
                    misses += 1
            else:
                misses += 1
        else:
            l = d['mention_position']
            keys = list(filter(lambda i: i > l - 2 and i < l + 2, d['spots'].keys()))
            v = {key: d['spots'][key] for key in keys}
            print({'mention': d['mention'], 'v': v})
            misses += 1
    print(ms)
    print('hits', hits)
    print('misses', misses)


def spot_dataset(dataset, spotter):
    """
    Creates dataset spots
    :param spotter:  Spotter function
    :param dataset:
    :return:
    """
    spotted_dataset = []
    c = 0
    for d in dataset:
        text = d['context_left'] + ' ' + d['mention'] + ' ' + d['context_right']
        d['spots'] = spotter(text.lower())  # <<<<<< !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        spotted_dataset.append(d)
    return spotted_dataset


def pckl(path, data):
    with open(path, 'wb') as f:
        print('Saving', path)
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def evaluate(dataset):
    """

    :param dataset:
    :return:
    """
    m = 0
    m_sum = 0
    m_max = 0
    hit = 0
    hit_first = 0
    miss = 0
    no_spot = 0
    i=0
    for d in dataset:
        mention = d['mention']#.lower()
        m += 1
        m_max = len(mention.split()) if len(mention.split()) > m_max else m_max
        m_sum += len(mention.split())
        if mention in spot_map:
            # print (spot_map[mention])
            # return
            # i+=1
            # if i%100==0:
            #     print (mention, type(commonness[mention]['id']), type(d['Wikipedia_ID']))

            if int(d['Wikipedia_ID']) == int(spot_map[mention]['id']):
                hit += 1
            else:
                miss+=1
                # if mention in commonness and int(commonness[mention]['id'])==int(d['Wikipedia_ID']):
                #     hit += 1
                # else:
                #     miss+=1
            # if d['Wikipedia_ID'] in spot_map[mention][0]:
            #     hit_first += 1
        else:
            no_spot += 1
            # if mention in commonness and int(commonness[mention]['id']) == int(d['Wikipedia_ID']):
            #     hit += 1
            # else:
            #
            #     miss += 1
            # print(d)

    print('hit', hit)
    print('miss ', miss)
    print('no_spot', no_spot)
    print('---')
    print('m_max', m_max)
    print('msum ', m_sum)
    print('m', m)
    print('m_avg', float(m_sum) / float(m))


def evaluate_run_time(dataset, method):
    """

    :param dataset:
    :return:
    """
    methodology = 'SR_norm'
    import time
    start = time.time()
    for d in dataset:
        text = (d['context_left'] + ' ' + d['mention'] + ' ' + d['context_right']).lower()
        spots = fuzzy_spotter(text)
        for s in spots.values():
            for e in s.values():
                o = spot_map.get(e['anchor_text'], {methodology: 0.0})[methodology]
    timing = time.time() - start
    print ("--- -%s seconds --" % timing)

if __name__ == '__main__':
    # for method in ['commonness', 'relative_commonness']:
    #     print(method, '\ntesta:')
    #     evaluate_run_time(testa, method)
    #     print('testb:')
    #     evaluate_run_time(testb, method)
    evaluate_run_time(testb,  'relative_commonness')
    evaluate_run_time(testa,  'relative_commonness')
    # spotted_dataset = spot_dataset(testb, spotter)
    #
    # for method in ['commonness', 'relative_commonness']:
    #     y_true, probs = link_and_evaluate_spotted_dataset(spotted_dataset, method)
    #     pckl('results/'+method+'_y_true', y_true)
    #     pckl('results/'+method+'_probs', probs)
    # spotted_dataset = spot_dataset(testb, total_spotter)
    # # for method in ['commonness', 'relative_commonness']:
    # #     y_true, probs = link_and_evaluate_spotted_dataset(spotted_dataset, method)
    # #     pckl('results/total_'+method+'_y_true', y_true)
    # #     pckl('results/total_'+method+'_probs', probs)
    # fuzzy_spotted_dataset = spot_dataset(testb, fuzzy_spotter)
    # for method in ['commonness', 'relative_commonness']:
    #     y_true, probs = link_and_evaluate_fuzzy_spotted_dataset(fuzzy_spotted_dataset, method)
    #     pckl('results/fuzzy_'+method+'_y_true', y_true)
    #     pckl('results/fuzzy_'+method+'_probs', probs)

    #
    # # print ('-0----------------------',spotted_dataset[0])
    # print('evaluating spotter')
    # # evaluate_spotter(spotted_dataset)
    # # evaluate(testa)
    # y_true_m1, probs_m1 = link_and_evaluate_spotted_dataset(spotted_dataset, 'commonness')
    # pckl('results/m1_y_true', y_true_m1)
    # pckl('results/m1_probs', probs_m1)
