import main
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

train2001 = main.trainData2001
train2016 = main.trainData2016

train2001 = train2001.drop(["'unit_id '", "'site '", "'posOAST '", "'dir '"], axis=1)
train2016 = train2016.drop(["Unit_id", "site", "dir"], axis=1)

main.clean(train2001)

train2001_labels = train2001['label']
train2016_labels = train2016['label']

x_train2001 = pd.DataFrame(train2001,
                           columns=['Speed', 'leftMeas', 'rightMeas', 'nomWeight','minMeas', 'maxMeas',
                                    'diffMeas', 'avgMeas'])
x_train2016 = pd.DataFrame(train2016,
                           columns=['Speed', 'leftMeas', 'rightMeas', 'nomWeight','minMeas', 'maxMeas',
                                    'diffMeas', 'avgMeas'])

y_train2001 = train2001_labels
y_train2016 = train2016_labels

features2001 = x_train2001.columns
features2016 = x_train2016.columns


# feature selection by Random Forest

def forest_selector(x_train, y_train, features):
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(x_train, y_train)
    plt.barh(features, rf.feature_importances_)
    plt.xlabel("Random Forest Feature Importance")
    plt.show()


def lassoSelector(x_train, y_train, features):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso())
    ])

    search = GridSearchCV(pipeline,
                          {'model__alpha': np.arange(0.1, 10, 0.1)},
                          cv=5, scoring="neg_mean_squared_error", verbose=3
                          )

    search.fit(x_train2001, y_train2001)
    coefficients = search.best_estimator_.named_steps['model'].coef_
    importance = np.abs(coefficients)

    print(np.array(features2001)[importance > 0])


# feature scaling to fit MLP model
sc = StandardScaler()
scaler2001 = sc.fit(x_train2001)
x_train2001 = scaler2001.transform(x_train2001)

scaler2016 = sc.fit(x_train2016)
x_train2016 = scaler2016.transform(x_train2016)

# feature selection calls
#forest_selector(x_train2001, y_train2001, features2001)
#lassoSelector(x_train2001, y_train2001, features2001)

forest_selector(x_train2016, y_train2016, features2016)
lassoSelector(x_train2016, y_train2016, features2016)
