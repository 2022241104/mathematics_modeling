import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = pd.read_csv('../data/Wimbledon_featured_matches(1).csv',
                   usecols=['p1_score', 'p2_score', 'server', 'point_victor',
                            'p1_games', 'p2_games', 'p1_break_pt_won', 'p2_break_pt_won', 'p1_points_won', 'p2_points_won'])

lost_rate = 1-data.count()/len(data)


session = pd.read_csv(
    '../data/Wimbledon_featured_matches(1).csv', usecols=['match_id'])

session = session['match_id'].str.split(
    '-', expand=True).drop([0, 1], axis=1)

session = session.astype(float)


score_mapping = {'0': 0, '15': 15, '30': 30, '40': 40,
                 'AD': 50, '1': 15, '2': 30, '3': 41.32, '4': 41.32, '5': 41.32, '6': 41.32, '7': 41.32, '8': 41.32, '9': 41.32, '10': 41.32, '11': 41.32, '12': 41.32, '13': 41.32}
data['p1_score'] = data['p1_score'].map(score_mapping)
data['p2_score'] = data['p2_score'].map(score_mapping)
data = pd.get_dummies(data, columns=['server', 'point_victor'])

Lost_ = 1-data.count()/len(data)


data['momentum'] = data['point_victor_1'].rolling(window=4).sum()  # 动量

data['impetus'] = data['p1_points_won'] - data['p2_points_won']  # 势头
data['momentum'][0] = 0
data['momentum'][1] = 0
data['momentum'][2] = 0
server_mapping = {True: 1, False: 0}
victor_mapping = {True: 1, False: 0}
data['server_1'] = data['server_1'].map(server_mapping)
data['server_2'] = data['server_2'].map(server_mapping)
data['point_victor_1'] = data['point_victor_1'].map(victor_mapping)
data['point_victor_2'] = data['point_victor_2'].map(victor_mapping)


X = data.drop(['point_victor_1', 'point_victor_2'], axis=1)
y = data['point_victor_1']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


accuracy = [0 for _ in range(3)]

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

rf_y_pred = model_rf.predict(X_test)


accuracy[0] = accuracy_score(y_test, rf_y_pred)


model_gbm = GradientBoostingClassifier(n_estimators=100, random_state=42)
model_gbm.fit(X_train, y_train)

gbm_y_pred = model_gbm.predict(X_test)

accuracy[1] = accuracy_score(y_test, gbm_y_pred)


estimators = [('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
              ('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=42)))]
clf = StackingClassifier(estimators=estimators,
                         final_estimator=LogisticRegression())

clf.fit(X_train, y_train)

clf_pre = clf.predict(X_test)

accuracy[2] = accuracy_score(y_test, clf_pre)


data.to_csv('../data/after.csv', index=False)
