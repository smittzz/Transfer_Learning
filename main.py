import pandas as pd
import sklearn as sk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import wasserstein_distance

# Data preprocessing tasks
test2001 = pd.read_csv('C:/Users/Bradyn/PycharmProjects/research/2001/2001/testDataset.csv')
train2001 = pd.read_csv('C:/Users/Bradyn/PycharmProjects/research/2001/2001/traindataset.csv')

test2016 = pd.read_csv('C:/Users/Bradyn/PycharmProjects/research/2016/2016/testDT.csv')
train2016 = pd.read_csv('C:/Users/Bradyn/PycharmProjects/research/2016/2016/trainDT.csv')


# method to remove filler characters and change data types of dataset
def clean(dataset):
    dataset.columns = dataset.columns.str.replace("[']", "")
    dataset.columns = dataset.columns.str.replace("[ ]", "")
    dataset.columns.str.strip()
    dataset['maxMeasSide'].replace(['Left', 'Righ'], [0, 1], inplace=True)  # if left then 0, if right then 1
    # dataset['posOAST'].replace(['before', 'after'], [0, 1], inplace=True)  # if before then 0, if after then 1
    # dataset['unit_id'] = train2001['unit_id'].str.strip("DTTX")  # remove DTTX prefix leaving unit_id as num
    # dataset['site'].replace(['BOLT', 'CARS', 'CPGO', 'GOGV', 'GRAN', 'GUEL', 'MORT', 'POPL', 'REDW', 'TBAY'],
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
    # dataset['dir'].replace(['N', 'E', 'S', 'W'], [0, 1, 2, 3], inplace=True)

    for c in dataset:
        dataset[c].replace(['?'], [0], inplace=True)

    for c in dataset:
        if dataset[c].dtype != 'float':
            dataset[c] = pd.to_numeric(dataset[c], downcast="float")

    return dataset


# create 1d plot of a given feature from two datasets
def plotMaker(dataset1, dataset2, ftr):
    plotFirst = pd.DataFrame(dataset1, columns=[ftr, 'label'])
    plotLast = pd.DataFrame(dataset2, columns=[ftr, 'label'])
    plotLast['label'].replace([0.0, 1.0], [2.0, 3.0], inplace=True)  # 0 is now labelled as 2, 1 labelled as 3
    plotFirst0 = plotFirst[plotFirst.label == 0]
    plotFirst1 = plotFirst[plotFirst.label == 1]
    plotLast0 = plotLast[plotLast.label == 2]
    plotLast1 = plotLast[plotLast.label == 3]

    sns.kdeplot(data=plotFirst0, x=ftr, color="navy", common_norm=False, label="2001 Pass")
    sns.kdeplot(data=plotFirst1, x=ftr, color="cyan", common_norm=False, label="2001 Fail", linestyle=(0, (5, 1)))

    sns.kdeplot(data=plotLast0, x=ftr, color="forestgreen", common_norm=False, label="2016 Pass")
    sns.kdeplot(data=plotLast1, x=ftr, color="lime", common_norm=False, label="2016 Fail",
                linestyle=(0, (5, 1)))

    plt.legend()
    plt.title(ftr + " " + "Density Distribution")
    plt.savefig(ftr + '.pdf')
    plt.clf()


# create 2d plot of given features from two datasets
def twoDPlot(dataset1, dataset2, ftr1, ftr2):
    plotFirst = pd.DataFrame(dataset1, columns=[ftr1, ftr2, 'label'])
    plotLast = pd.DataFrame(dataset2, columns=[ftr1, ftr2, 'label'])
    plotLast['label'].replace([0.0, 1.0], [2.0, 3.0], inplace=True)  # 0 is now labelled as 2, 1 labelled as 3
    plotFirst0 = plotFirst[plotFirst.label == 0]
    plotFirst1 = plotFirst[plotFirst.label == 1]
    plotLast0 = plotLast[plotLast.label == 2]
    plotLast1 = plotLast[plotLast.label == 3]

    sns.kdeplot(data=plotFirst0, x=ftr1, y=ftr2, color="darkviolet", common_norm=False, label="2001 Pass", fill=False,
                linewidths=0.5
                )
    sns.kdeplot(data=plotFirst1, x=ftr1, y=ftr2, color="cyan", common_norm=False, label="2001 Fail", fill=False,
                linewidths=0.5)
    sns.kdeplot(data=plotLast0, x=ftr1, y=ftr2, color="orange", common_norm=False, label="2016 Pass", fill=False,
                linewidths=0.5
                )
    sns.kdeplot(data=plotLast1, x=ftr1, y=ftr2, color="lime", common_norm=False, label="2016 Fail", fill=False,
                linewidths=0.5)

    plt.legend()
    plt.title(ftr1 + " " + "+" + " " + ftr2 + " " + "Density Distribution")
    plt.savefig(ftr1 + "_" + ftr2 + '.pdf')
    plt.clf()


# helper method to create 1d plot of all features
def massPlot():
    plotMaker(train2001, train2016, 'Speed')
    plotMaker(train2001, train2016, 'leftMeas')
    plotMaker(train2001, train2016, 'rightMeas')
    plotMaker(train2001, train2016, 'nomWeight')
    plotMaker(train2001, train2016, 'minMeas')
    plotMaker(train2001, train2016, 'maxMeas')
    plotMaker(train2001, train2016, 'diffMeas')
    plotMaker(train2001, train2016, 'avgMeas')


# build t-SNE from given two labels (1 dataset)
def createTSNE(label0, label1, title):
    tsne_em = TSNE(n_components=2, perplexity=40.0, random_state=0)
    data_label0 = tsne_em.fit_transform(label0)
    data_label1 = tsne_em.fit_transform(label1)
    plt.figure(figsize=(6, 6))
    line1 = plt.scatter(data_label0[:, 0], data_label0[:, 1], s=7)
    line2 = plt.scatter(data_label1[:, 0], data_label1[:, 1], s=7)
    plt.legend((line1, line2), ("Pass " + title, "Fail " + title), loc='upper right')
    plt.savefig("tsne_ " + title + ".pdf")
    plt.clf()


# build t_SNE from given four labels (2 datasets)
def comboTSNE(label0, label1, label2, label3):
    tsne_em = TSNE(n_components=2, perplexity=40.0, random_state=0)
    data_label0 = tsne_em.fit_transform(label0)
    data_label1 = tsne_em.fit_transform(label1)
    data_label2 = tsne_em.fit_transform(label2)
    data_label3 = tsne_em.fit_transform(label3)
    plt.figure(figsize=(6, 6))
    line1 = plt.scatter(data_label0[:, 0], data_label0[:, 1], s=7)
    line2 = plt.scatter(data_label1[:, 0], data_label1[:, 1], s=7)
    line3 = plt.scatter(data_label2[:, 0], data_label2[:, 1], s=7)
    line4 = plt.scatter(data_label3[:, 0], data_label3[:, 1], s=7)
    plt.legend((line1, line2, line3, line4), ("Pass 2001", "Fail 2001", "Pass 2016", "Fail 2016"), loc='upper right')
    plt.savefig("tsne_combined.pdf")
    plt.clf()


# get wasserstein distance between 2001 and 2016 for given feature
def getDistance(x, y, ftr):
    x = x[ftr]
    y = y[ftr]
    d = 0
    i = wasserstein_distance(x, y)
    minn = i
    j = wasserstein_distance(x, y - (d - 0.01))

    for n in range(2000):
        i = wasserstein_distance(x, y - d)

        if i > j:
            d = d - 0.01
        if i < j:
            d = d + 0.01

        if i < minn:
            minn = i
    return minn, d


train2001 = train2001.drop(["'unit_id '", "'site '", "'posOAST '", "'dir '"], axis=1)
train2016 = train2016.drop(["Unit_id", "site", "dir"], axis=1)

clean(train2001)
clean(train2016)

# train2016 = train2016.dropna()

# min-max scaling on features
scaler = sk.preprocessing.MinMaxScaler()
names2001 = train2001.columns
names2016 = train2016.columns
scale2001 = scaler.fit_transform(train2001)
scale2016 = scaler.fit_transform(train2016)
scaled_2001 = pd.DataFrame(scale2001, columns=names2001)
scaled_2016 = pd.DataFrame(scale2016, columns=names2016)
label0 = scaled_2001[scaled_2001.label == 0]
label1 = scaled_2001[scaled_2001.label == 1]
label2 = scaled_2016[scaled_2016.label == 0]
label3 = scaled_2016[scaled_2016.label == 1]
# label0 = label0.append(label2, ignore_index=True)
# label1 = label1.append(label3, ignore_index=True)

# tSNE call
# createTSNE(label0, label1, "2001")
# createTSNE(label2, label3, "2016")
# comboTSNE(label0,label1,label2,label3)

# feature-wise distribution visualizations
# massPlot()

# twoDPlot(train2001,train2016,'Speed','maxMeasSide')

distArr = getDistance(train2001, train2016, 'avgMeas')
distArrTwo = getDistance(train2001, train2016, 'Speed')
minNum = distArr[0]
minNumTwo = distArrTwo[0]
d = distArr[1]
dTwo = distArrTwo[1]
train2016['avgMeas'] = train2016['avgMeas'] - d
train2016['Speed'] = train2016['Speed'] - dTwo

#plotMaker(train2001, train2016, 'avgMeas')
twoDPlot(train2001, train2016, 'Speed', 'avgMeas')

train2016Shift = train2016
