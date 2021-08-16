# -*- coding: utf-8 -*-
"""
Author:ZUO XINYU
Data:29/07/2017
"""
import Params
import dataProcess_trigger_train
import numpy as np
import random


def get_tensor(positive_sample_path, negative_sample_path, sentence_length, positive_sample_num):
    """
    Get positive and negative sample tensor
    :param positive_sample_path:
    :param negative_sample_path:
    :return: positive and negative sample
    """
    sample_word_train = []
    sample_sentence_train = []
    sample_position_train = []
    label_train = []
    sample_adjacent_words_train = []
    sample_parts_indexs_train = []
    sample_flags_train = []

    # sample_word_test = []
    # sample_sentence_test = []
    # sample_position_test = []
    # label_test = []
    # sample_adjacent_words_test = []
    # sample_parts_indexs_test = []
    # sample_flags_test = []

    num_positive = 0
    num_negative = 0

    temp1 = 0
    temp2 = 0

    positive_sample_word = []
    positive_sample_sentence = []
    positive_sample_position = []
    positive_sample_label = []
    positive_sample_adjacent_words = []
    positive_sample_flags = []

    print("Get all positive sample...")
    with open(positive_sample_path, 'r', encoding='utf-8') as positive_sample_file:
        for line in positive_sample_file.readlines():
            sentence_temp = []
            position_temp = []
            label_temp = []
            adjacent_temp = []
            num_positive += 1
            line.strip()
            temp1 += 1
            if num_positive % 10 == 1:
                positive_sample_word.append(int(line))
                positive_sample_flags.append(int(1))
            if num_positive % 10 == 3:
                sentence_id = line.split()
                for id in sentence_id:
                    sentence_temp.append(int(id))
                positive_sample_sentence.append(sentence_temp)
            if num_positive % 10 == 5:
                position_id = line.split()
                for id in position_id:
                    position_temp.append(int(int(id) + int(79)))
                positive_sample_position.append(position_temp)
            if num_positive % 10 == 7:
                label_id = line.split()
                for id in label_id:
                    label_temp.append(int(id))
                positive_sample_label.append(label_temp)
            if num_positive % 10 == 9:
                adjacent_id = line.split()
                for id in adjacent_id:
                    adjacent_temp.append(int(id))
                positive_sample_adjacent_words.append(adjacent_temp)

            # if temp1 == 50:
            #     break

    print("Positive sample all %d" % len(positive_sample_word))

    print("Get all negative sample...")
    negative_sample_word = []
    negative_sample_sentence = []
    negative_sample_position = []
    negative_sample_label = []
    negative_sample_adjacent_words = []
    negative_sample_flags = []
    with open(negative_sample_path, 'r', encoding='utf-8') as negative_sample_file:
        for line in negative_sample_file.readlines():
            sentence_temp = []
            position_temp = []
            label_temp = []
            adjacent_temp = []
            num_negative += 1
            line.strip()
            temp2 += 1
            if num_negative % 10 == 1:
                negative_sample_word.append(int(line))
                negative_sample_flags.append(int(0))
            if num_negative % 10 == 3:
                sentence_id = line.split()
                for id in sentence_id:
                    sentence_temp.append(int(id))
                negative_sample_sentence.append(sentence_temp)
            if num_negative % 10 == 5:
                position_id = line.split()
                for id in position_id:
                    position_temp.append(int(int(id) + int(79)))
                negative_sample_position.append(position_temp)
            if num_negative % 10 == 7:
                label_id = line.split()
                for id in label_id:
                    label_temp.append(int(id))
                negative_sample_label.append(label_temp)
            if num_negative % 10 == 9:
                adjacent_id = line.split()
                for id in adjacent_id:
                    adjacent_temp.append(int(id))
                negative_sample_adjacent_words.append(adjacent_temp)
            if num_negative == 15 * num_positive:
                break

            # if temp2 == 50:
            #     break
    print("Negative sample all %d" % len(negative_sample_label))

    print("Shuffle positive sample...")
    combine_positive_sample = list(
        zip(positive_sample_word, positive_sample_sentence, positive_sample_position, positive_sample_label,
            positive_sample_adjacent_words, positive_sample_flags))
    random.shuffle(combine_positive_sample)
    positive_sample_word, positive_sample_sentence, positive_sample_position, positive_sample_label, positive_sample_adjacent_words, positive_sample_flags = zip(*combine_positive_sample)

    print("Shuffle negative sample...")
    combine_negative_sample = list(
        zip(negative_sample_word, negative_sample_sentence, negative_sample_position, negative_sample_label,
            negative_sample_adjacent_words, negative_sample_flags))
    random.shuffle(combine_negative_sample)
    negative_sample_word, negative_sample_sentence, negative_sample_position, negative_sample_label, negative_sample_adjacent_words, negative_sample_flags = zip(*combine_negative_sample)

    print("Get positive for train and test...")
    # positive_sample_word_test = []
    # positive_sample_sentence_test = []
    # positive_sample_position_test = []
    # positive_sample_label_test = []
    # positive_sample_adjacent_words_test = []
    # positive_sample_flags_test = []
    for i in range(len(positive_sample_label)):
        if i < positive_sample_num:
            sample_word_train.append(positive_sample_word[i])
            sample_sentence_train.append(positive_sample_sentence[i])
            sample_position_train.append(positive_sample_position[i])
            label_train.append(positive_sample_label[i])
            sample_adjacent_words_train.append(positive_sample_adjacent_words[i])
            sample_flags_train.append(positive_sample_flags[i])
        # else:
        #     positive_sample_word_test.append(positive_sample_word[i])
        #     positive_sample_sentence_test.append(positive_sample_sentence[i])
        #     positive_sample_position_test.append(positive_sample_position[i])
        #     positive_sample_label_test.append(positive_sample_label[i])
        #     positive_sample_adjacent_words_test.append(positive_sample_adjacent_words[i])
        #     positive_sample_flags_test.append(positive_sample_flags[i])
    positive_sample_num = len(label_train)
    print("Positive sample for train %d" % len(label_train))
    # print("Positive sample for test %d" % len(positive_sample_label_test))

    print("Get negative for train and test...")
    # negative_sample_word_test = []
    # negative_sample_sentence_test = []
    # negative_sample_position_test = []
    # negative_sample_label_test = []
    # negative_sample_adjacent_words_test = []
    # negative_sample_flags_test = []
    print("Negative sample for train %d" % i)
    for i in range(len(negative_sample_word)):
        if i < 15 * positive_sample_num:
            sample_word_train.append(negative_sample_word[i])
            sample_sentence_train.append(negative_sample_sentence[i])
            sample_position_train.append(negative_sample_position[i])
            label_train.append(negative_sample_label[i])
            sample_adjacent_words_train.append(negative_sample_adjacent_words[i])
            sample_flags_train.append(negative_sample_flags[i])
        # elif 15 * positive_sample_num <= i < 15 * positive_sample_num + 8453 * 15:
        #     negative_sample_word_test.append(negative_sample_word[i])
        #     negative_sample_sentence_test.append(negative_sample_sentence[i])
        #     negative_sample_position_test.append(negative_sample_position[i])
        #     negative_sample_label_test.append(negative_sample_label[i])
        #     negative_sample_adjacent_words_test.append(negative_sample_adjacent_words[i])
        #     negative_sample_flags_test.append(negative_sample_flags[i])
        else:
            # print("Negative sample for test %d" % len(negative_sample_label))
            break

    # for i in range(len(positive_sample_label_test)):
    #     sample_word_test.append(positive_sample_word[i])
    #     sample_sentence_test.append(positive_sample_sentence[i])
    #     sample_position_test.append(positive_sample_position[i])
    #     label_test.append(positive_sample_label[i])
    #     sample_adjacent_words_test.append(positive_sample_adjacent_words[i])
    #     sample_flags_test.append(positive_sample_flags[i])
    #
    # for i in range(len(negative_sample_label_test)):
    #     sample_word_test.append(negative_sample_word[i])
    #     sample_sentence_test.append(negative_sample_sentence[i])
    #     sample_position_test.append(negative_sample_position[i])
    #     label_test.append(negative_sample_label[i])
    #     sample_adjacent_words_test.append(negative_sample_adjacent_words[i])
    #     sample_flags_test.append(negative_sample_flags[i])

    print("All data for train %d" % len(label_train))
    # print("All data for test %d" % len(label_test))

    # sample_parts_indexs_test = get_trigger_parts_index(sample_position_test, sentence_length)
    # sample_test = list(
    #     zip(sample_word_test, sample_sentence_test, sample_position_test, label_test,
    #         sample_parts_indexs_test, sample_adjacent_words_test, sample_flags_test))

    # write_test_sample_into_file("./data/sample_test_0809_1.tsv", sample_test)

    sample_parts_indexs_train = get_trigger_parts_index(sample_position_train, sentence_length)

    # print(sample_word_train)
    # print(sample_sentence_train)
    # print(sample_position_train)
    # print(label_train)
    # print(smaple_adjacent_words_train)
    # print(sample_parts_indexs)
    # print(sample_parts_indexs.shape)

    return sample_word_train, sample_sentence_train, sample_position_train, label_train, sample_parts_indexs_train, \
           sample_adjacent_words_train, sample_flags_train


def get_trigger_parts_index(sample_position_train, sentence_length):
    """
    Get trigger parts index
    :param sample_position_train:
    :return: Sample trigger parts index list
    """
    print("Get trigger parts index...")
    sample_parts_indexs = []
    for sample_position in sample_position_train:
        part_1 = [float(1.0)] * sentence_length
        part_2 = [float(1.0)] * sentence_length
        part_combine = []
        num = 0
        for sample_position_id in sample_position:
            if int(sample_position_id) == 79:
                part_1[:num] = [float(1.0)] * (num + 1)
                part_1[num + 1:] = [float(0.0)] * (sentence_length - num - 1)
                part_2[:num] = [float(0.0)] * (num + 1)
                part_2[num + 1:] = [float(1.0)] * (sentence_length - num - 1)
            num += 1
        part_1 = np.array(part_1)
        part_2 = np.array(part_2)
        part_combine.append(part_1)
        part_combine.append(part_2)
        part_combine = np.array(part_combine)
        sample_parts_indexs.append(part_combine)
    # sample_parts_indexs = np.array(sample_parts_indexs)

    print("Get sample done!")
    return sample_parts_indexs


def write_test_sample_into_file(filename, combine_sample_test):
    """
    Write test sample into file
    :param filename:
    :return:
    """
    sample_word_test, sample_sentence_test, sample_position_test, label_test, sample_parts_indexs_test, \
    sample_adjacent_words_test, sample_flags_test = zip(*combine_sample_test)

    print("Write test data into file...")
    with open(filename, 'w', encoding='utf-8') as file:
        for i in range(len(sample_word_test)):
            file.write(str(sample_word_test[i]))
            file.write('\n')
            for word in sample_sentence_test[i]:
                file.write(str(word) + " ")
            file.write('\n')
            for position in sample_position_test[i]:
                file.write(str(position) + " ")
            file.write('\n')
            for label in label_test[i]:
                file.write(str(label) + " ")
            file.write('\n')
            for adjacent_words in sample_adjacent_words_test[i]:
                file.write(str(adjacent_words) + " ")
            file.write('\n')
            file.write(str(sample_flags_test[i]) + " ")
            file.write('\n')
    file.close()
    print("Write test data into file done!")


if __name__ == '__main__':
    get_tensor(Params.positive_sample_path, Params.negative_sample_path, Params.sentence_length)