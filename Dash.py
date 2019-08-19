import csv
import pandas as pd
import TCRC
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from itertools import cycle
from sklearn import svm
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.base import clone
from sklearn.decomposition import PCA, KernelPCA

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings

warnings.filterwarnings('ignore')

# 以下方法专为Dash数据集设计 #

# 提取特征 #
def statisticsKmer(epiDict, k=3):
    kmerDict = {}
    for epi in epiDict:
        for i in range(len(epiDict[epi])):
            splitList = TCRC.splitCDR(epiDict[epi][i], k)
            for split in splitList:
                if split not in kmerDict:
                    kmerDict[split] = 1
                else:
                    kmerDict[split] += 1
    return kmerDict


def buildFeatures(epiDict, kmerDict, k=3):
    counter = 0
    for epi in epiDict:
        counter += len(epiDict[epi])
    retArr = np.zeros((counter, len(kmerDict)))

    kmerList = kmerDict.keys()
    retLabel = []

    iter = 0
    epinum = 0
    for epi in epiDict:
        for cdr in range(len(epiDict[epi])):
            splitlist = TCRC.splitCDR(epiDict[epi][cdr], k)
            retLabel.append(epinum)
            i = 0
            for kmer in kmerList:
                retArr[iter][i] = splitlist.count(kmer)
                i += 1
            iter += 1
        epinum += 1
    return np.array(retArr), np.array(retLabel)

# end 提取特征 #

# 以下方法用于绘制ROC曲线 #

def _cal_micro_ROC(y_test, y_score):
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())
    return fpr, tpr, auc(fpr, tpr)


def _cal_macro_ROC(y_test, y_score, fpr, tpr, n_classes):
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    return all_fpr, mean_tpr, auc(all_fpr, mean_tpr)


def _cal_roc_auc(y_test, y_score, y_pred, n_classes, draw_roc_curve, title):
    fpr = dict()
    tpr = dict()
    precision = list()
    recall = list()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(roc_auc[i])
        precision.append(precision_score(y_test[:, i], y_pred[:, i]))
        recall.append(recall_score(y_test[:, i], y_pred[:, i]))

    # micro-average ROC
    fpr["micro"], tpr["micro"], roc_auc["micro"] = _cal_micro_ROC(y_test, y_score)

    # macro-average ROC
    fpr["macro"], tpr["macro"], roc_auc["macro"] = _cal_macro_ROC(y_test, y_score, fpr, tpr, n_classes)

    # plot all ROC curves
    if draw_roc_curve:
        _plot_roc_curves2(fpr, tpr, roc_auc, n_classes, title)

    return roc_auc, np.mean(precision), np.mean(recall)

def _plot_roc_curves2(fpr, tpr, roc_auc, n_classes, title):
    mean_fpr = np.linspace(0, 1, 200)
    tprs = list()
    aucs = list()
    for i in range(n_classes):
        tprs.append(interp(mean_fpr, fpr[i], tpr[i]))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc[i])
        plt.plot(fpr[i],tpr[i],lw=1,alpha=0.5,label=epiname_list[i])

    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='b', label='Mean ROC', lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.show()

def _plot_roc_curves(fpr, tpr, roc_auc, n_classes, title):
    plt.plot(fpr["micro"], tpr["micro"], label='micro ROC ({0:0.3f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=2)
    plt.plot(fpr["macro"], tpr["macro"], label='macro ROC ({0:0.3f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=2)
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=1, label=epiname_list[i] + ' ({0:0.3f})'.format(roc_auc[i]))
        # plt.plot(fpr[i], tpr[i], lw=1, label=' ({0:0.3f})'.format(roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="best")
    plt.savefig(title)
    plt.show()

def predict_auc(X, y, classifier, cv, class_number, draw_roc_curve, title):
    y = label_binarize(y, classes=np.arange(class_number))

    n_classes = y.shape[1]
    n_samples, n_features = X.shape

    kf = KFold(n_splits=cv, shuffle=True, random_state=666)
    auc_dict = {}
    acc_sum = 0
    precision_sum = 0
    recall_sum = 0
    cur_fold = 1
    for train_index, test_index in kf.split(X, y):
        print('=' * (10 * cur_fold) + '>' + '-' * (10 * (cv - cur_fold)))
        # copy classifier
        clone_clf = clone(classifier)

        # split cross-validation folds
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        # model evaluation
        clf = clone_clf.fit(X_train, y_train)
        # y_score = clf.decision_function(X_test)
        y_prob = clf.predict_proba(X_test)

        # plt.subplot(121)
        # sns.heatmap(y_prob)
        # plt.xlabel('Class')
        # plt.ylabel('Predicted TCR epitope')
        # plt.subplot(122)
        # sns.heatmap(y_test)
        # plt.xlabel('Class')
        # plt.ylabel('Real TCR epitope')
        # plt.show()

        y_pred = clf.predict(X_test)
        cur_acc = clf.score(X_test, y_test)
        auc_dict[cur_fold], precision, recall = _cal_roc_auc(y_test, y_prob, y_pred, class_number, draw_roc_curve,
                                          title + '_fold' + str(cur_fold) + '.png')
        print("### Fold-{0:d} ###" .format(cur_fold))
        print("ACC: {0:f}".format(cur_acc))
        print("Precision: {0:f}".format(precision))
        print("Recall: {0:f}".format(recall))
        # print("Micro AUC: {0:f}".format(auc_dict[cur_fold]['micro']))
        print("Macro AUC: {0:f}".format(auc_dict[cur_fold]['macro']))

        acc_sum = acc_sum + cur_acc
        precision_sum = precision_sum + precision
        recall_sum = recall_sum + recall
        cur_fold = cur_fold + 1

    print("cross-validation Finished!")
    return auc_dict, acc_sum / cv, precision_sum / cv, recall_sum / cv

# end 绘制ROC曲线 #

# PCA
def pca_analyse(X, n_components):
    pca = PCA(n_components=n_components)
    newX = pca.fit_transform(X)
    print("PCA analysing ...  {0:} features remained".format(n_components))

    """
       将数据保存为文件
       """
    storage_path = 'experiment3/pca_data_' + str(n_components) + '.csv'
    print(newX.shape)

    with open(storage_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in range(len(newX)):
            row = list(newX[i])
            row.append(y[i])
            writer.writerow(row)

    print("原始数据存储完成")

    return newX

def kernel_pca_analyse(X, n_components, kernel):
    pca = KernelPCA(n_components=n_components, kernel=kernel)
    newX = pca.fit_transform(X)
    print("PCA analysing ...  {0:} features remained".format(n_components))

    return newX

# end PCA

def data_preprocess(file, k):
    df = pd.read_csv(file)

    epiDict = {}
    epiname_list = []
    for index, row in df.iterrows():
        if row[0] not in epiDict:
            epiDict[row[0]] = []
        epiDict[row[0]].append(row[1])

    # 筛选可用数据
    # Threshold：10
    epiDict_filtered = {}
    for epi in epiDict:
        if len(epiDict[epi]) > 10:
            epiDict_filtered[epi] = epiDict[epi]
    epiDict = epiDict_filtered

    # 剔除重复数据
    # 可删除此项
    for epi in epiDict:
        epiname_list.append(epi)
        # cdr3_list = []
        # for cdr in epiDict[epi]:
        #     if cdr not in cdr3_list:
        #         cdr3_list.append(cdr)
        # epiDict[epi] = cdr3_list

    statistics_epi = []
    statistics_num = []
    print('{:22s} {:s}'.format('Epitope', 'Number'))
    for epi in epiDict:
        statistics_epi.append(epi)
        statistics_num.append(len(epiDict[epi]))
        print('{:22s} {:d}'.format(epi, len(epiDict[epi])))

    # 绘制统计图
    # sns.barplot(statistics_num, statistics_epi)
    # plt.show()

    kmerDict = statisticsKmer(epiDict, k)

    """
    提取kmer特征
    """
    print("正在提取kmer特征...")
    X, y = buildFeatures(epiDict, kmerDict, k)
    print("数据建立完毕")

    """
    将数据保存为文件
    """
    storage_path = 'experiment3/selected_data_origin.csv'
    print(X.shape)

    with open(storage_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        head = list(kmerDict)
        head.append('class')
        writer.writerow(head)
        for i in range(len(X)):
            row = list(X[i])
            row.append(y[i])
            writer.writerow(row)

    print("PCA降维数据存储完成")
    return X, y, epiname_list

def plot_pca_2D(X, k):
    node_list = {}

    newX = pca_analyse(X, k)
    for i in range(len(y)):
        if not y[i] in node_list:
            node_list[y[i]] = []
        node_list[y[i]].append({"x": newX[i][0], "y": newX[i][1]})

    for item in node_list:
        x_list = []
        y_list = []
        for node in node_list[item]:
            x_list.append(node['x'])
            y_list.append(node['y'])

        plt.scatter(x_list, y_list)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

    plt.legend()
    plt.show()

def plot_kernel_pca_2D(X, k):
    kernel_list = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
    node_list = {}

    for i in range(5):
        plt.subplot(1,5,i+1)
        newX = kernel_pca_analyse(X, k, kernel=kernel_list[i])
        for i in range(len(y)):
            if not y[i] in node_list:
                node_list[y[i]] = []
            node_list[y[i]].append({"x": newX[i][0], "y": newX[i][1]})

        for item in node_list:
            x_list = []
            y_list = []
            for node in node_list[item]:
                x_list.append(node['x'])
                y_list.append(node['y'])

            plt.scatter(x_list, y_list)
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')

    plt.show()

# 绘制二维可视化图
def plot_2D(X, y):
    node_list = {}

    # newX = pca_analyse(X, 2)
    newX = SelectKBest(chi2, k=2).fit_transform(X, y)
    for i in range(len(y)):
        if not y[i] in node_list:
            node_list[y[i]] = []
        node_list[y[i]].append({"x": newX[i][0], "y": newX[i][1]})

    for item in node_list:
        x_list = []
        y_list = []
        for node in node_list[item]:
            x_list.append(node['x'])
            y_list.append(node['y'])

        plt.scatter(x_list, y_list)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

    plt.legend()
    plt.show()

def plot_3D(X, y):
    from mpl_toolkits.mplot3d import Axes3D
    node_list = {}

    newX = pca_analyse(X, 3)
    for i in range(len(y)):
        if not y[i] in node_list:
            node_list[y[i]] = []
        node_list[y[i]].append({"x": newX[i][0], "y": newX[i][1], "z": newX[i][2]})

    for item in node_list:
        x_list = []
        y_list = []
        z_list = []
        for node in node_list[item]:
            x_list.append(node['x'])
            y_list.append(node['y'])
            z_list.append(node['z'])

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(x_list, y_list, z_list)

    plt.show()

# 绘制热力图
def draw_heatmap(X, y):
    print('ploting heatmap ...')
    sns.heatmap(np.c_[X, y], cmap=sns.cm.rocket_r)
    plt.show()



print('Train and Test ...')
mean_micro_auc = []
mean_macro_auc = []
acc_list = []
# classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', C=4, gamma=0.125))
# classifier = OneVsRestClassifier(GradientBoostingClassifier())
classifier = OneVsRestClassifier(GradientBoostingClassifier(learning_rate=0.1,
                                                            min_samples_leaf=20, max_features='sqrt', subsample=0.8,
                                                            random_state=10, n_estimators=150, max_depth=11,
                                                            min_samples_split=100))
# classifier = OneVsRestClassifier(RandomForestClassifier())


X, y, epiname_list = data_preprocess('data/Dash/Dash_mouse.csv', 1)
# X, y = load_iris().data, load_iris().target
# newX = SelectKBest(chi2, k=20).fit_transform(X, y)
newX = pca_analyse(X, 0.9)
# newX = X
# kernel_list = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
# newX = kernel_pca_analyse(X, 20, 'poly')
# draw_heatmap(newX, y)

roc_auc, acc, precision, recall = predict_auc(newX, y, classifier, cv=5, class_number=7, draw_roc_curve=True, title='experiment3/Dash_all_' + str(X.shape[1]) + '_GBDT_cv5_')
mean_macro_auc.append(round(pd.DataFrame(roc_auc).loc["macro"].mean(), 3))
mean_micro_auc.append(round(pd.DataFrame(roc_auc).loc["micro"].mean(), 3))
# print(mean_micro_auc)
print("mean_macro_auc: {0:}".format(mean_macro_auc))
print("precision: {0:}".format(precision))
print("recall: {0:}".format(recall))
print("acc: {0:}".format(acc))
draw_heatmap(newX*10, y)

# for i in range(int(1900 / 10)):
#     print("Training model: {0:}".format(i + 1))
#     newX = pca_analyse(X, 10 * (i + 1))
#
#     # 在这里修改关键参数
#     roc_auc, acc = predict_auc(newX, y, classifier, cv=5, class_number=10, draw_roc_curve=False, title='experiment3/Dash_all_' + str(newX.shape[1]) + '_GBDT_cv5_500_')
#     mean_macro_auc.append(round(pd.DataFrame(roc_auc).loc["macro"].mean(), 3))
#     mean_micro_auc.append(round(pd.DataFrame(roc_auc).loc["micro"].mean(), 3))
#     acc_list.append(acc)
#     print("acc: ", acc_list)
#     print("mean_micro_auc: ", mean_micro_auc)
#     print("mean_macro_auc: ", mean_macro_auc)
#     # print('ploting heatmap ...')
#     # sns.heatmap(np.c_[newX, y*0.2], cmap=sns.cm.rocket_r)
#     # plt.show()