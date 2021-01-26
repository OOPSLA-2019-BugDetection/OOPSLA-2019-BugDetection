from configparser import ConfigParser
import numpy as np
import sklearn as sk


def set_threshold(input_list, threshold):
    fixed_list = []
    for i in range(len(input_list)):
        if input_list[i] > threshold:
            fixed_list.append(1)
        else:
            fixed_list.append(0)
    return fixed_list


cfg = ConfigParser()
cfg.read('config.ini')
evaluation_data = cfg.get('globalcontext', 'test_evaluation_data')
predicted_data = cfg.get('globalcontext', 'test_output_data')
evaluation_d = np.load(evaluation_data)
predicted_d = np.load(predicted_data)
y_true = evaluation_d.flatten()
y_pred = predicted_d.flatten()
recall = 0
precision = 0
f_score = 0
for i in range(len(y_pred)):
    y_pred_fix = set_threshold(y_pred, y_pred[i])
    recall_temp = sk.metrics.precision_score(y_true, y_pred_fix)
    precision_temp = sk.metrics.recall_score(y_true, y_pred_fix)
    f_score_temp = sk.metrics.f1_score(y_true, y_pred_fix)
    if f_score_temp>f_score:
        f_score = f_score_temp
        recall = recall_temp
        precision = precision_temp
print("F Score: " + f_score)
print("Precision: " + precision)
print("Recall: " + recall)
