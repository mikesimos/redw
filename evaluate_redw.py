import json
import os
import pickle
from etl_dataset import testa, testb

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
        "anchor_id": None,
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
                d['Wikipedia_ID'] == spot_map.get(d['mention'], {'id': None})['id']:
            true_prediction += [1]
            probabilities += [spot_map[d['mention']][methodology]]
            # if spot_map.get(anchor, {methodology: 0.0})[methodology] == None:
            #     print ('GOT NONE----')
            #     print (methodology, anchor)
            #     print (spot_map.get(anchor))
        else:

            true_prediction += [0]
            spoted_mention_position = sorted(list(filter(lambda i: i <= d['mention_position'], d['spots'].keys())))[-1]
            anchor = d['spots'][spoted_mention_position]['anchor_text']
            # if spot_map.get(anchor, {methodology: 0.0})[methodology] == None:
            #     print ('GOT NONE')
            #     print (methodology, anchor)
            #     print (spot_map.get(anchor))
            probabilities += [spot_map.get(anchor, {methodology: 0.0})[methodology]]
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
    i =0
    for d in spotted_dataset:
        if i>0:
            i-=1
            print (d)
        if d['mention_position'] in d['spots'] and \
                d['spots'][d['mention_position']]['length'] == d['mention_length']:
            if d['spots'][d['mention_position']].get('anchor_id'):
                if d['spots'][d['mention_position']].get('anchor_id')['id']==d['Wikipedia_ID']:
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
        d['spots'] = spotter(text)  # <<<<<< !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        spotted_dataset.append(d)
    return spotted_dataset


def pckl(path, data):
    with open(path, 'wb') as f:
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
    for d in dataset:
        mention = d['mention']#.lower()
        m += 1
        m_max = len(mention.split()) if len(mention.split()) > m_max else m_max
        m_sum += len(mention.split())
        if mention in spot_map:
            redw_spotter()
            # print (spot_map[mention])
            # return
            hit+=1
            # if d['Wikipedia_ID'] == spot_map[mention]['id']:
            #     hit += 1
            # else:
            #     miss+=1
            # if d['Wikipedia_ID'] in spot_map[mention][0]:
            #     hit_first += 1
        else:
            no_spot += 1
            # print(d)
            miss += 1

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
        spots = redw_spotter(text)
        for s in spots.values():
            o = spot_map.get(s['anchor_text'], {methodology: 0.0})[methodology]
    timing = time.time() - start
    print ("--- -%s seconds --" % timing)

if __name__ == '__main__':
    # spotted_dataset = redw_spot_dataset(testb, redw_spotter)
    # evaluate(testa)
    # evaluate_spotter(spotted_dataset)
    for method in ['SR_norm', 'SR_min_max_norm']:
        print(method, '\ntesta:')
        evaluate_run_time(testa, method)
        print('testb:')
        evaluate_run_time(testb, method)
        # y_true_a, probs_a = redw_link_and_evaluate_spotted_dataset(spotted_dataset, method)
        # pckl('results/'+method+'_y_true', y_true_a)
        # pckl('results/'+method+'_probs', probs_a)
#
