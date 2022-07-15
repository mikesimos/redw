import matplotlib.pyplot as plt
import pickle
import numpy as np
FIGURE_N = 1

def get_f1_curve(precision, recall):
    f_measure = []
    for r, p in zip(recall, precision):
        f_measure.append(2*(p*r)/(p+r))
    return f_measure


def get_f1_curve_table(precision, recall, f1_measure):
    l = len(precision)
    print('recall\tprec\tf1')
    for i in np.arange(0.1, 1.1, 0.1):
        index = int(l*i)-1
        print("%.4f %.4f %.4f" % (recall[index], precision[index], f1_measure[index]))


results = {}
for dataset in ['AIDA-YAGO2-testa', 'AIDA-YAGO2-testb', 'Clueweb', 'WNEDWiki']:
    results[dataset] = {}
    for method in ['redw_SR_norm', 'redw_SR_min_max_norm', 'relative_commonness', 'fuzzy_relative_commonness']:
        with open('results/' + dataset + '_' + method+'_y_true', 'rb') as fp1:
            with open('results/' + dataset + '_' + method + '_probs', 'rb') as fp2:
                results[dataset][method] = {
                    'y_true': pickle.load(fp1),
                    'probs': pickle.load(fp2),
                }


def get_pr_curve(predictions, probabilities):
    zipped = list(zip(predictions, probabilities))
    pairs = sorted(zipped, key=lambda item: item[1], reverse=True)
    predictions, probabilities = list(zip(*pairs))
    precision_y = []
    recall_x = []
    l = len(predictions)
    for i in np.arange(0.005, 1.005, 0.005):
        precision_y.append(sum(predictions[0:int(i*l)])/float(int(i*l)))
        recall_x.append(i)
    return recall_x, precision_y


def plot_dataset(dataset='AIDA-YAGO2-testa'):
    # Precision - Recall
    global FIGURE_N
    plt.figure(FIGURE_N)
    FIGURE_N += 1
    plt.title(dataset + ' (Precision - Recall)')
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    for method, color in zip(['relative_commonness', 'redw_SR_norm', 'redw_SR_min_max_norm', 'fuzzy_relative_commonness'],
                             ['blue', 'orange', 'pink', 'green']):
        rec, prec = get_pr_curve(results[dataset][method]['y_true'], results[dataset][method]['probs'])
        plt.plot(rec, prec, color=color, label=method, lw=2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.show()

    # F1 - Recall
    plt.figure(FIGURE_N)
    FIGURE_N += 1
    plt.title(dataset + ' (F1 Score - Recall)')
    for method, color in zip(['relative_commonness', 'redw_SR_norm', 'redw_SR_min_max_norm', 'fuzzy_relative_commonness'],
                             ['blue', 'orange', 'pink', 'green']):
        rec, prec = get_pr_curve(results[dataset][method]['y_true'], results[dataset][method]['probs'])
        f1 = get_f1_curve(prec, rec)
        print(method)
        get_f1_curve_table(prec, rec, f1)
        plt.plot(rec, f1, color=color, label=method, lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('F1 Score')
    plt.legend(loc='upper left')
    plt.show()


plot_dataset('AIDA-YAGO2-testa')
plot_dataset('AIDA-YAGO2-testb')
plot_dataset('Clueweb')
plot_dataset('WNEDWiki')