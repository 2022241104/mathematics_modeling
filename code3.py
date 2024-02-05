# %%
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
from datetime import timedelta
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('../data/Wimbledon_featured_matches(2).csv')
data = data.drop(['p1_games', 'p2_games', 'match_id', 'player1', 'player2', 'set_no', 'p1_score', 'p2_score',
                 'point_no', 'p1_sets', 'p2_sets', 'game_no', 'p1_points_won', 'p2_points_won', 'speed_mph'], axis=1)

# %%

lost_rate = 1-data.count()/len(data)
data = pd.get_dummies(data, columns=['server', 'point_victor', 'serve_no',
                      'winner_shot_type', 'serve_width', 'serve_depth', 'return_depth'])
data['momentum'] = data['point_victor_1'].rolling(window=3).sum()  # 动量
mapping = {True: 1, False: 0}
data['server_1'] = data['server_1'].map(mapping)
data['server_2'] = data['server_2'].map(mapping)
data['serve_no_1'] = data['serve_no_1'].map(mapping)
data['serve_no_2'] = data['serve_no_2'].map(mapping)
data['winner_shot_type_B'] = data['winner_shot_type_B'].map(mapping)
data['winner_shot_type_F'] = data['winner_shot_type_F'].map(mapping)
data['serve_width_B'] = data['serve_width_B'].map(mapping)
data['serve_width_BC'] = data['serve_width_BC'].map(mapping)
data['serve_width_BW'] = data['serve_width_BW'].map(mapping)
data['serve_width_C'] = data['serve_width_W'].map(mapping)
data['serve_width_W'] = data['serve_width_W'].map(mapping)
data['serve_depth_CTL'] = data['serve_depth_CTL'].map(mapping)
data['serve_depth_NCTL'] = data['serve_depth_NCTL'].map(mapping)
data['return_depth_D'] = data['return_depth_D'].map(mapping)
data['return_depth_ND'] = data['return_depth_ND'].map(mapping)

# %%
data['elapsed_time'] = pd.to_datetime(data['elapsed_time'])
data['time_difference'] = data['elapsed_time'].diff()
data['time_difference_in_seconds'] = data['time_difference'].apply(
    lambda x: x.total_seconds())
for i in range(len(data)):
    if data['time_difference_in_seconds'][i] < 0:
        data['time_difference'][i] = 0
data['time_difference_in_seconds'] = (data['time_difference_in_seconds'] -
                                      data['time_difference_in_seconds'].mean())/data['time_difference_in_seconds'].std()
data['time_difference_in_seconds'][0] = 0
data['p1_distance_run'] = (data['p1_distance_run'] -
                           data['p1_distance_run'].mean())/data['p1_distance_run'].std()
data['p2_distance_run'] = (data['p2_distance_run'] -
                           data['p2_distance_run'].mean())/data['p2_distance_run'].std()
data['rally_count'] = (data['rally_count'] -
                       data['rally_count'].mean()) / data['rally_count'].std()

data = data.drop(['time_difference', 'elapsed_time', 'point_victor_1',
                 'point_victor_2', 'winner_shot_type_0'], axis=1)

data['momentum'][0] = 0
data['momentum'][1] = 0
# data['momentum'][2] = 0
data['momentum_vary'] = data['momentum'].diff()
data['momentum_vary'][0] = 0
data = data.drop(['momentum'], axis=1)
# data['momentum'] = (data['momentum'] - data['momentum'].mean())/data['momentum'].std()
# data['speed_mph'] = (data['speed_mph'] - data['speed_mph'].mean())/data['speed_mph'].std()
# data['rally_count'] = (data['rally_count'] - data['rally_count'].mean())/data['rally_count'].std()
# data['p1_distance_run'] = (data['p1_distance_run'] - data['p1_distance_run'].mean())/data['p1_distance_run'].std()
# data['p2_distance_run'] = (data['p2_distance_run'] - data['p2_distance_run'].mean())/data['p2_distance_run'].std()

# %%

X = data.drop('momentum_vary', axis=1)
y = data['momentum_vary']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# %%

model_rf = RandomForestClassifier(
    n_estimators=300, random_state=42, max_depth=8,)
# param =param = {"n_estimators":[120,200,300,500,800,1200], "max_depth":[5,8,15,25,30]}
# gc = GridSearchCV(model_rf, param_grid=param, cv=4)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
model_rf.fit(X_train, y_train)
# gc.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:", classification_rep)
print("Confusion Matrix:", confusion_mat)

# print(gc.best_params_)

# %%

feature_importances_rf = pd.Series(
    model_rf.feature_importances_, index=X.columns)
feature_importances_rf.nlargest(20).plot(kind='barh')

# %%

y_prob = model_rf.predict_proba(X_test)[:, 1]
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_prob)), y_prob, alpha=0.6,
            label='Predicted Probability of Momentum Change')
plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold for Prediction')
plt.xlabel('Point Index')
plt.ylabel('Probability of Momentum Change')
plt.title('Predicted Probability of Momentum Change Over Points')
plt.legend()
plt.show()

# %%
model_gbm = GradientBoostingClassifier(
    n_estimators=300, random_state=42, max_depth=8)
param = {"n_estimators": [120, 200, 300, 500,
                          800, 1200], "max_depth": [5, 8, 15, 25, 30]}
gc = GridSearchCV(model_rf, param_grid=param, cv=4)
# gc.fit(X_train, y_train)
model_gbm.fit(X_train, y_train)
# print(gc.best_params_)
y_pred = model_gbm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:", classification_rep)
print("Confusion Matrix:", confusion_mat)

# %%
feature_importances_gbm = pd.Series(
    model_gbm.feature_importances_, index=X.columns)
feature_importances_gbm.nlargest(20).plot(kind='barh')

# %%
y_prob = model_rf.predict_proba(X_test)[:, 1]
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_prob)), y_prob, alpha=0.6,
            label='Predicted Probability of Momentum Change')
plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold for Prediction')
plt.xlabel('Point Index')
plt.ylabel('Probability of Momentum Change')
plt.title('Predicted Probability of Momentum Change Over Points')
plt.legend()
plt.show()

# %%
estimators = [('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
              ('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=42)))]
model_clf = StackingClassifier(estimators=estimators,
                               final_estimator=LogisticRegression())
# param =param = {"n_estimators":[120,200,300,500,800,1200], "max_depth":[5,8,15,25,30]}
# gc = GridSearchCV(model_clf, param_grid=param, cv=4)
# gc.fit(X_train, y_train)
# print(gc.best_params_)
model_clf.fit(X_train, y_train)
# y_pred = model_clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# classification_rep = classification_report(y_test, y_pred)
# confusion_mat = confusion_matrix(y_test, y_pred)
# print("Accuracy:", accuracy)
# print("Classification Report:", classification_rep)
# print("Confusion Matrix:", confusion_mat)


# %%
# base_estimators = model_clf.get_params()['estimators']

# base_estimator = base_estimators[0][1]
# feature_importances_clf = base_estimator.feature_importances_
# feature_importances_clf.nlargest(20).plot(kind='barh')

# %%
y_prob = model_rf.predict_proba(X_test)[:, 1]
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_prob)), y_prob, alpha=0.6,
            label='Predicted Probability of Momentum Change')
plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold for Prediction')
plt.xlabel('Point Index')
plt.ylabel('Probability of Momentum Change')
plt.title('Predicted Probability of Momentum Change Over Points')
plt.legend()
plt.show()

# %%
# 对森林和决策树训练的特征重要性取均值
feature_importances = (feature_importances_gbm + feature_importances_rf)/2
feature_importances.nlargest(20).plot(kind='barh')

# %%
data = pd.read_excel('../data/2023-wimbledon-女单.xlsx')
data = data.drop(['p1_games', 'p2_games', 'match_id', 'player1', 'player2', 'set_no', 'p1_score', 'p2_score',
                 'point_no', 'p1_sets', 'p2_sets', 'game_no', 'p1_points_won', 'p2_points_won', 'speed_mph'], axis=1)
lost_rate = 1-data.count()/len(data)
data = pd.get_dummies(data, columns=['server', 'point_victor', 'serve_no',
                      'winner_shot_type', 'serve_width', 'serve_depth', 'return_depth'])
data['momentum'] = data['point_victor_1'].rolling(window=3).sum()  # 动量
mapping = {True: 1, False: 0}
data['server_1'] = data['server_1'].map(mapping)
data['server_2'] = data['server_2'].map(mapping)
data['serve_no_1'] = data['serve_no_1'].map(mapping)
data['serve_no_2'] = data['serve_no_2'].map(mapping)
data['winner_shot_type_F'] = data['winner_shot_type_F'].map(mapping)
data['serve_width_B'] = data['serve_width_B'].map(mapping)
data['serve_width_BC'] = data['serve_width_BC'].map(mapping)
data['serve_width_BW'] = data['serve_width_BW'].map(mapping)
data['serve_width_C'] = data['serve_width_W'].map(mapping)
data['serve_width_W'] = data['serve_width_W'].map(mapping)
data['serve_depth_CTL'] = data['serve_depth_CTL'].map(mapping)
data['serve_depth_NCTL'] = data['serve_depth_NCTL'].map(mapping)
data['return_depth_D'] = data['return_depth_D'].map(mapping)
data['return_depth_ND'] = data['return_depth_ND'].map(mapping)

# %%
# from datetime import datetime, time
# t = time(data['elapsed_time'])
# data['elapsed_time'] = datetime.combine(datetime.now(),data['elapsed_time'])
data['time_difference_in_seconds'] = data['elapsed_time'].diff()
for i in range(len(data)):
    if data['time_difference_in_seconds'][i] < 0:
        data['time_difference_in_seconds'][i] = 0
data['time_difference_in_seconds'] = (data['time_difference_in_seconds'] -
                                      data['time_difference_in_seconds'].mean())/data['time_difference_in_seconds'].std()
data['time_difference_in_seconds'][0] = 0
data['p1_distance_run'] = (data['p1_distance_run'] -
                           data['p1_distance_run'].mean())/data['p1_distance_run'].std()
data['p2_distance_run'] = (data['p2_distance_run'] -
                           data['p2_distance_run'].mean())/data['p2_distance_run'].std()
data['rally_count'] = (data['rally_count'] -
                       data['rally_count'].mean()) / data['rally_count'].std()
# 将用过的这些特征删除
data = data.drop(['elapsed_time', 'point_victor_1',
                 'point_victor_2', 'winner_shot_type_0'], axis=1)
# 定义动量变化
data['momentum'][0] = 0
data['momentum'][1] = 0
# data['momentum'][2] = 0
data['momentum_vary'] = data['momentum'].diff()
data['momentum_vary'][0] = 0
data = data.drop(['momentum'], axis=1)
X = data.drop('momentum_vary', axis=1)
y = data['momentum_vary']

# %%
y_pred = model_rf.predict(X)
accuracy = accuracy_score(y, y_pred)
classification_rep = classification_report(y, y_pred)
confusion_mat = confusion_matrix(y, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:", classification_rep)
print("Confusion Matrix:", confusion_mat)

# %%
y_prob = model_rf.predict_proba(X)[:, 1]
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_prob)), y_prob, alpha=0.6,
            label='Predicted Probability of Momentum Change')
plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold for Prediction')
plt.xlabel('Point Index')
plt.ylabel('Probability of Momentum Change')
plt.title('Predicted Probability of Momentum Change Over Points')
plt.legend()
plt.show()

# %%
y_pred = model_gbm.predict(X)
accuracy = accuracy_score(y, y_pred)
classification_rep = classification_report(y, y_pred)
confusion_mat = confusion_matrix(y, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:", classification_rep)
print("Confusion Matrix:", confusion_mat)

# %%
y_prob = model_gbm.predict_proba(X)[:, 1]
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_prob)), y_prob, alpha=0.6,
            label='Predicted Probability of Momentum Change')
plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold for Prediction')
plt.xlabel('Point Index')
plt.ylabel('Probability of Momentum Change')
plt.title('Predicted Probability of Momentum Change Over Points')
plt.legend()
plt.show()

# %%
y_pred = model_clf.predict(X)
accuracy = accuracy_score(y, y_pred)
classification_rep = classification_report(y, y_pred)
confusion_mat = confusion_matrix(y, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:", classification_rep)
print("Confusion Matrix:", confusion_mat)

# %%
y_prob = model_clf.predict_proba(X)[:, 1]
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_prob)), y_prob, alpha=0.6,
            label='Predicted Probability of Momentum Change')
plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold for Prediction')
plt.xlabel('Point Index')
plt.ylabel('Probability of Momentum Change')
plt.title('Predicted Probability of Momentum Change Over Points')
plt.legend()
plt.show()
