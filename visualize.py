import matplotlib.pyplot as plt
import pickle
import numpy as np


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
for method in ['REDW_SR_norm', 'REDW_SR_min_max_norm', 'relative_commonness', 'fuzzy_relative_commonness']:
    with open('results/'+method+'_y_true', 'rb') as fp1:
        with open('results/' + method + '_probs', 'rb') as fp2:
            results[method] = {
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
    for i in np.arange(0.001, 1.001, 0.001):
        precision_y.append(sum(predictions[0:int(i*l)])/float(int(i*l)))
        recall_x.append(i)
    return recall_x, precision_y


plt.figure()

f_scores = np.linspace(0.2, 0.8, num=4)
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))


# print(sorted(results['SR_norm']['probs']))
# print(len(results['SR_norm']['y_true']))
for method, color in zip(['REDW_SR_norm', 'REDW_SR_min_max_norm', 'relative_commonness', 'fuzzy_relative_commonness'], ['orange', 'pink', 'blue', 'green']):

    rec, prec = get_pr_curve(results[method]['y_true'], results[method]['probs'])
    f1 = get_f1_curve(prec, rec)
    print(method)
    get_f1_curve_table(prec, rec, f1)
    plt.plot(rec, f1, color=color, label=method, lw=2)

# redw_sr_norm_rec, redw_sr_norm_prec = get_pr_curve(redw_sr_norm_y_true, redw_sr_norm_probs)
# redw_sr_norm_f1 = get_f1_curve(redw_sr_norm_prec, redw_sr_norm_rec)
# print('RedW SR Norm')
# get_f1_curve_table(redw_sr_norm_prec, redw_sr_norm_rec, redw_sr_norm_f1)
#
#
# plt.plot(redw_sr_norm_rec, redw_sr_norm_prec, color='green', label="redw SR Norm", lw=2)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('F1 Score')
plt.legend(loc='upper left')
plt.show()

# fig2 = plt.figure()
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('Recall')
# plt.ylabel('F1 Score')
# plt.legend('redw SR F1',loc='upper left')
# plt.plot(redw_sr_rec, redw_sr_f1, color='orange', label="redw SR F1", lw=2)
# plt.plot(redw_sr_norm_rec, redw_sr_norm_f1, color='green', label="redw SR F1", lw=2)
# plt.show()
