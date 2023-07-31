import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def format_conversion(tl, pl):
    tru, pre = [], []

    for i in range(len(tl)):
        for j in range(len(tl[i])):
            pre.append(int(pl[i][j]))
            tru.append(int(tl[i][j]))

    return tru, pre


def evaluation(tl, pl, eps=1e-6):
    tp, fn, fp, tn = 0, 0, 0, 0
    tru, pre = format_conversion(tl, pl)

    for i in range(len(tru)):
        if tru[i] == 0 and pre[i] == 0: tp += 1
        if tru[i] == 1 and pre[i] == 1: tn += 1
        if tru[i] == 1 and pre[i] == 0: fp += 1
        if tru[i] == 0 and pre[i] == 1: fn += 1

    acc = (tp + tn) / (tn + fp + tp + fn + eps)
    npv = tn / (tn + fn + eps)
    ppv = tp / (tp + fp + eps)
    sen = tp / (tp + fn + eps)
    spe = tn / (tn + fp + eps)
    fos = 2 * (ppv * sen) / (ppv + sen + eps)

    return acc, npv, ppv, sen, spe, fos


def confusion(con_str, tl, pl):
    tru, pre = format_conversion(tl, pl)
    conf_mat = confusion_matrix(tru, pre)
    df_cm = pd.DataFrame(conf_mat, index=['positive', 'negative'], columns=['positive', 'negative'])
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='center')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig('./recording/' + str(con_str) + 'matrix.png', bbox_inches='tight', pad_inches=0.0, dpi=300)
    plt.show()
