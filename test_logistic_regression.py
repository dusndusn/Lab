import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

x_train_logistic=np.load('x_train_logistic.npy')
x_test_logistic=np.load('x_test_logistic.npy')
y_train=np.load('y_train.npy')
y_test=np.load('y_test.npy')

print(x_train_logistic.shape)
print(x_test_logistic.shape)
print(y_train.shape)
print(y_test.shape)



model=LogisticRegression(max_iter=10000)
model.fit(x_train_logistic, y_train)
joblib.dump(model, 'logistic_regression_model.pkl')

y_train_pred=model.predict_proba(x_train_logistic)[:, 1]
train_auroc=roc_auc_score(y_train, y_train_pred)
train_auprc=average_precision_score(y_train, y_train_pred)

y_test_pred=model.predict_proba(x_test_logistic)[:, 1]
test_auroc=roc_auc_score(y_test, y_test_pred)
test_auprc=average_precision_score(y_test, y_test_pred)

student_id='20215044'
with open(f'{student_id}_logistic_regression.txt', 'w') as f:
    f.write(f'{student_id}\n')
    f.write(f'{train_auroc:.4f}\n')
    f.write(f'{train_auprc:.4f}\n')
    f.write(f'{test_auroc:.4f}\n')
    f.write(f'{test_auprc:.4f}\n')

