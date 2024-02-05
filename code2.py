# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import chisquare
data = pd.read_csv('../data/after.csv')
data['momentum'] = data['point_victor_1'].rolling(window=3).sum()
data['momentum'][0] = 0
data['momentum'][1] = 0
# data['momentum'][2] = 0

# %%
fact = data['momentum']
predict = pd.Series(np.random.choice(
    [0, 1, 2, 3], size=len(data)))

# %%
fact_array = fact.value_counts().sort_index().values
predict_array = predict.value_counts().sort_index().values
chi_square_stat, p_value = chisquare(fact_array, predict_array)
print(f"Chi-Square Statistic: {chi_square_stat}, P-value: {p_value}")

# %%
fact_and_predict = pd.DataFrame({'fact': fact, 'predict': predict})
fact_and_predict_array = fact_and_predict.value_counts().sort_index()
print(fact_and_predict_array)

# %%

y_true = fact.values
y_pred = predict.values

roc_points = {}
for threshold in range(100):
    threshold /= 100
    y_pred_binary = [int(pred >= threshold) for pred in y_pred]
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred_binary):
        if yt and yp:
            tp += 1
        elif not yt and yp:
            fp += 1
        elif not yt and not yp:
            tn += 1
        else:
            fn += 1
    roc_points[threshold] = (tp / (tp + fn), fp / (fp + tn))

roc_x = [point[1] for point in sorted(roc_points.items())]
roc_y = [point[0] for point in sorted(roc_points.items())]
plt.plot(roc_x, roc_y)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
