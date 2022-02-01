import main
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

test2001 = pd.read_csv('C:/Users/Bradyn/PycharmProjects/research/2001/2001/testDataset.csv')
train2001 = pd.read_csv('C:/Users/Bradyn/PycharmProjects/research/2001/2001/traindataset.csv')

test2016 = pd.read_csv('C:/Users/Bradyn/PycharmProjects/research/2016/2016/testDT.csv')
train2016 = pd.read_csv('C:/Users/Bradyn/PycharmProjects/research/2016/2016/trainDT.csv')

train2001 = train2001.drop(["'unit_id '", "'site '", "'posOAST '", "'dir '"], axis=1)
test2001 = test2001.drop(["'unit_id '", "'site '", "'posOAST '", "'dir '"], axis=1)
train2016 = train2016.drop(["Unit_id", "site", "dir"], axis=1)
test2016 = test2016.drop(["Unit_id", "site", "Dir"], axis=1)

main.clean(train2001)
main.clean(test2001)

train2001Final = pd.DataFrame(train2001, columns=['Speed', 'leftMeas', 'rightMeas', 'nomWeight', 'minMeas', 'maxMeas',
                                                  'diffMeas', 'avgMeas'])
test2001Final = pd.DataFrame(test2001, columns=['Speed', 'leftMeas', 'rightMeas', 'nomWeight', 'minMeas', 'maxMeas',
                                                'diffMeas', 'avgMeas'])
train2016Final = pd.DataFrame(train2016, columns=['Speed', 'leftMeas', 'rightMeas', 'nomWeight', 'minMeas', 'maxMeas',
                                                  'diffMeas', 'avgMeas'])
test2016Final = pd.DataFrame(test2016, columns=['Speed', 'leftMeas', 'rightMeas', 'nomWeight', 'minMeas', 'maxMeas',
                                                  'diffMeas', 'avgMeas'])

train2001_labels = train2001['label']
test2001_labels = test2001['class']
train2016_labels = train2016['label']
test2016_labels = test2016['label']

x_train2001 = train2001Final
y_train2001 = train2001_labels
x_test2001 = test2001Final
y_test2001 = test2001_labels

x_train2016 = train2016Final
y_train2016 = train2016_labels
x_test2016 = test2016Final
y_test2016 = test2016_labels

sc = StandardScaler()
scaler2001 = sc.fit(x_train2001)
x_train2001 = scaler2001.transform(x_train2001)
x_test2001 = scaler2001.transform(x_test2001)

scaler2016 = sc.fit(x_train2016)
x_train2016 = scaler2001.transform(x_train2016)
x_test2016 = scaler2001.transform(x_test2016)


def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements


clf = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(15,5), random_state=1, alpha=1e-4,
                    learning_rate_init=0.001,
                    max_iter=300, verbose=10)

clf.fit(x_train2016, y_train2016)

print("Training set score: %f" % clf.score(x_train2016, y_train2016))

y_pred = clf.predict(x_test2016)
cm = confusion_matrix(y_pred, y_test2016)

print("Accuracy of Classification Prediction: ", accuracy(cm))
