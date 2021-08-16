# -*- coding: utf-8 -*-

"""
Author:ZUO XINYU
Data:16/08/2017
"""
import Params
import dataProcess_trigger_bag
import numpy as np
import random


def get_trigger_bag_tensor(sentence_filename):
    """
    Get trigger bag sample
    :return: Trigger bag sample
    """
    trigger_list, key_argument_list, trigger_type_list = dataProcess_trigger_bag.get_trigger_and_key_argument(
        sentence_filename, Params.trigger_type_path)
    positive_sentence_bag, negative_sentence_bag, positive_sentence_bag_trigger_type_vector, negative_sentence_bag_trigger_type_vector = dataProcess_trigger_bag.get_sentence_bag(
        sentence_filename, trigger_type_list, key_argument_list)
    vocabulary_id_dict = dataProcess_trigger_bag.read_vocabulary(Params.argument_vocabulary_path)
    positive_sentence_word_bag, positive_sentence_word_bag_padding = dataProcess_trigger_bag.get_sentence_word_bag(positive_sentence_bag, Params.sentence_length, trigger_type_list)
    negative_sentence_word_bag, negative_sentence_word_bag_padding = dataProcess_trigger_bag.get_sentence_word_bag(negative_sentence_bag, Params.sentence_length, trigger_type_list)
    positive_word_position_list_bag, positive_sentence_pre_word_list_bag = dataProcess_trigger_bag.get_sentence_word_position_and_sentence_pre_word(
        positive_sentence_word_bag, positive_sentence_word_bag_padding)
    negative_word_position_list_bag, negative_sentence_pre_word_list_bag = dataProcess_trigger_bag.get_sentence_word_position_and_sentence_pre_word(
        negative_sentence_word_bag, negative_sentence_word_bag_padding)
    positive_sentence_pre_word_id_list_bag = dataProcess_trigger_bag.get_sentence_word_id_bag(vocabulary_id_dict, positive_sentence_pre_word_list_bag)
    negative_sentence_pre_word_id_list_bag = dataProcess_trigger_bag.get_sentence_word_id_bag(vocabulary_id_dict, negative_sentence_pre_word_list_bag)
    positive_adjacent_words_bag = dataProcess_trigger_bag.get_adjacent_words_bag(positive_sentence_word_bag)
    negative_adjacent_words_bag = dataProcess_trigger_bag.get_adjacent_words_bag(negative_sentence_word_bag)
    positive_adjacent_words_id_bag = dataProcess_trigger_bag.get_adjacent_words_id_bag(vocabulary_id_dict, positive_adjacent_words_bag)
    negative_adjacent_words_id_bag = dataProcess_trigger_bag.get_adjacent_words_id_bag(vocabulary_id_dict, negative_adjacent_words_bag)
    positive_trigger_parts_index_bag = dataProcess_trigger_bag.get_trigger_parts_index_bag(positive_word_position_list_bag, Params.sentence_length)
    negative_trigger_parts_index_bag = dataProcess_trigger_bag.get_trigger_parts_index_bag(negative_word_position_list_bag, Params.sentence_length)

    # print(positive_word_position_list_bag)
    # print(positive_trigger_parts_index_bag)

    positive_bag_flags = []
    negative_bag_flags = []

    for i in range(len(positive_sentence_word_bag)):
        positive_bag_flags.append(int(1))
    for i in range(len(negative_sentence_word_bag)):
        negative_bag_flags.append(int(0))

    word_list_bag = []
    sentence_pre_word_id_list_bag = []
    word_position_list_bag = []
    label_bag = []
    adjacent_words_id_bag = []
    trigger_parts_index_bag = []
    bag_flags = []

    word_list_bag.extend(positive_sentence_word_bag)
    word_list_bag.extend(negative_sentence_word_bag)
    sentence_pre_word_id_list_bag.extend(positive_sentence_pre_word_id_list_bag)
    sentence_pre_word_id_list_bag.extend(negative_sentence_pre_word_id_list_bag)
    word_position_list_bag.extend(positive_word_position_list_bag)
    word_position_list_bag.extend(negative_word_position_list_bag)
    label_bag.extend(positive_sentence_bag_trigger_type_vector)
    label_bag.extend(negative_sentence_bag_trigger_type_vector)
    adjacent_words_id_bag.extend(positive_adjacent_words_id_bag)
    adjacent_words_id_bag.extend(negative_adjacent_words_id_bag)
    trigger_parts_index_bag.extend(positive_trigger_parts_index_bag)
    trigger_parts_index_bag.extend(negative_trigger_parts_index_bag)
    bag_flags.extend(positive_bag_flags)
    bag_flags.extend(negative_bag_flags)

    return word_list_bag, sentence_pre_word_id_list_bag, word_position_list_bag, label_bag, adjacent_words_id_bag, trigger_parts_index_bag, bag_flags


def get_trigger_bag_pre_tensor(sentence_filename):
    """
    Get bag pre info
    :param sentence_filename:
    :return: Bag pre info
    """
    pos_pos_neg_bag_id_list, neg_pos_neg_bag_id_list, positive_bag_id_list, negative_bag_id_list = dataProcess_trigger_bag.get_bag_org_ids(sentence_filename, Params.trigger_type_path)
    trigger_list, key_argument_list, trigger_type_list = dataProcess_trigger_bag.get_trigger_and_key_argument(sentence_filename, Params.trigger_type_path)
    positive_sentence_bag, negative_sentence_bag, positive_sentence_bag_trigger_type_vector, negative_sentence_bag_trigger_type_vector = dataProcess_trigger_bag.get_sentence_bag(sentence_filename, trigger_type_list, key_argument_list)
    positive_sentence_word_num_bag = dataProcess_trigger_bag.get_sentence_word_num_bag(positive_sentence_bag, Params.sentence_length, trigger_type_list)
    negative_sentence_word_num_bag = dataProcess_trigger_bag.get_sentence_word_num_bag(negative_sentence_bag, Params.sentence_length, trigger_type_list)
    pos_neg_bag_org_ids = []
    bag_org_ids = []
    sentences_word_num = []
    bag_org_ids.extend(positive_bag_id_list)
    bag_org_ids.extend(negative_bag_id_list)
    pos_neg_bag_org_ids.extend(pos_pos_neg_bag_id_list)
    pos_neg_bag_org_ids.extend(neg_pos_neg_bag_id_list)
    sentences_word_num.extend(positive_sentence_word_num_bag)
    sentences_word_num.extend(negative_sentence_word_num_bag)

    return bag_org_ids, pos_neg_bag_org_ids, sentences_word_num


if __name__ == '__main__':
    # get_trigger_bag_tensor("./data/bag_temp.txt")
    get_trigger_bag_pre_tensor("./data/bag_temp.txt")