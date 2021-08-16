# -*- coding:utf-8 -*-

"""
Process trigger bag prediction result information.
Author:ZXY
Date:2017/11/29
"""
import Params
import numpy as np
import matplotlib.pyplot as plt


def extract_prediction_info(prediction_info_filename):
    """
    Extract prediction information.
    :param prediction_info_filename:
    :return: The result of extracted.
    """

    bag_org_ids = []
    pos_neg_bag_org_ids = []
    scores_max = []
    pre_labels = []
    org_labels = []
    scores_max_ids = []
    max_words = []
    scores_max_mul_ids = []
    with open(prediction_info_filename, 'r', encoding='utf-8') as prediction_info_file:
        for line in prediction_info_file.readlines():
            scores_max_mul_ids_pre_sentence = []
            line = line.strip()
            tokens = line.split()
            for i in range(len(tokens)):
                if i == 0:
                    bag_org_ids.append(int(tokens[i]))
                elif i == 1:
                    pos_neg_bag_org_ids.append(tokens[i])
                elif i == 2:
                    scores_max.append(float(tokens[i]))
                elif i == 3:
                    pre_labels.append(int(tokens[i]))
                elif i == 4:
                    org_labels.append(int(tokens[i]))
                elif i == 5:
                    scores_max_ids.append(int(tokens[i]))
                elif i == 6:
                    max_words.append(str(tokens[i]))
                else:
                    scores_max_mul_ids_pre_sentence.append(str(tokens[i]))
            scores_max_mul_ids.append(scores_max_mul_ids_pre_sentence)
        prediction_info_file.close()

    return bag_org_ids, pos_neg_bag_org_ids, scores_max, pre_labels, org_labels, scores_max_ids, max_words, scores_max_mul_ids


def sort_by_score(bag_org_ids, pos_neg_bag_org_ids, scores_max, pre_labels, org_labels, scores_max_ids, max_words, scores_max_mul_ids):
    """
    Sort by score.
    :param bag_org_ids: 
    :param pos_neg_bag_org_ids: 
    :param scores_max: 
    :param pre_labels: 
    :param org_labels: 
    :param scores_max_ids: 
    :param max_words: 
    :param scores_max_mul_ids: 
    :return: Sorted result.
    """
    predict_y = np.array(pre_labels)
    predict_y_prob = np.array(scores_max)
    y_given = np.array(org_labels)
    # print(predict_y)
    # print(predict_y_prob)
    # print(y_given)

    # sort prob
    index = np.argsort(predict_y_prob)[::-1]
    # print(index.shape[0])

    all_pre = [0.0]
    all_rec = [0.0]
    all_F1 = [0.0]
    positive_given_num = 0
    positive_prediction_num = 0
    positive_prediction_correct_num = 0
    for k in range(y_given.shape[0]):
        if y_given[index[k]] != 21:
            positive_given_num += 1
    for i in range(index.shape[0]):
        precision = 0.0
        recall = 0.0
        if predict_y[index[i]] != 21:
            positive_prediction_num += 1
        if y_given[index[i]] == predict_y[index[i]] and y_given[index[i]] != 21:
            positive_prediction_correct_num += 1
        if positive_prediction_num == 0:
            precision = 1.0
        elif positive_given_num == 0:
            recall = 1.0
        else:
            precision = float(positive_prediction_correct_num) / positive_prediction_num
            recall = float(positive_prediction_correct_num) / positive_given_num

        if precision != all_pre[-1] or recall != all_rec[-1]:
            all_pre.append(precision)
            all_rec.append(recall)
    for j in range(len(all_pre)):
        if j != 0:
            F1 = float(2.0 * all_pre[j] * all_rec[j]) / float(all_pre[j] + all_rec[j])
            all_F1.append(F1)
        else:
            all_F1.append(float(0.0))
    return all_pre, all_rec, all_F1


def plot(all_pre, all_rec, all_F1):
    """
    Plot P-R line.
    :param all_pre:
    :param all_rec:
    :param all_F1:
    :return:Plot P-R line.
    """
    recall_temp = []
    precision_temp = []
    for i in range(len(all_pre)):
        if all_pre[i] != 0.0 and all_pre[i] != 1.0 and all_rec[i] != 0.0 and all_rec[i] != 1.0:
            recall_temp.append(all_rec[i])
            precision_temp.append(all_pre[i])
    x = np.array(recall_temp)
    y = np.array(precision_temp)
    plt.figure(1)
    plt.plot(x, y)

    plt.title('DEV P-R')

    # plt.annotate("50", xy=(all_rec[49], all_pre[49]), xytext=(0, 0), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # my_x_ticks = np.arange(1.0, 0.01, 0.01)
    # my_y_ticks = np.arange(1.0, 0.01, 0.01)
    # plt.xticks(my_x_ticks)
    # plt.yticks(my_y_ticks)

    plt.show()


if __name__ == "__main__":
    bag_org_ids, pos_neg_bag_org_ids, scores_max, pre_labels, org_labels, scores_max_ids, max_words, scores_max_mul_ids = extract_prediction_info(Params.bag_dev_predict_info_path)
    precision_all, recall_all, F_value_all = sort_by_score(bag_org_ids, pos_neg_bag_org_ids, scores_max, pre_labels, org_labels, scores_max_ids, max_words, scores_max_mul_ids)
    # print(precision_recall_all[0][50])
    # print(precision_recall_all[1][50])
    # print(F_value_all)
    plot(precision_all, recall_all, F_value_all)
