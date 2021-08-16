# -*- coding:utf-8 -*-

"""
Tensor for argument input.
Author:ZXY
Date:2017/12/07
"""

import Params
import dataProcess_argument_train
import dataProacess_argument_val


def get_argument_bag_tensor(sentence_filename):
    """
    Get argument bag sample.
    :param sentence_filename:
    :return: Argument bag sample.
    """
    vocabulary_id_dict = dataProcess_argument_train.read_vocabulary(Params.argument_vocabulary_path)
    mid_vocabulary = dataProcess_argument_train.read_mid_vocabulary(Params.mid_vocabulary_path)
    trigger_list, key_argument_list, argument_type_list, trigger_type_list = dataProcess_argument_train.get_trigger_and_key_argument(sentence_filename, Params.argument_type_path, Params.trigger_type_path)
    positive_argument_sentence_bag, negative_argument_sentence_bag, positive_sentence_bag_argument_type_vector, \
    negative_sentence_bag_argument_type_vector, positive_argument_trigger_bag, negative_argument_trigger_bag, \
    positive_event_type_bag, negative_event_type_bag, positive_argument_mid_pre_bag, negative_argument_mid_pre_bag = dataProcess_argument_train.get_sentence_argument_bag(sentence_filename, argument_type_list, key_argument_list)
    positive_argument_sentence_word_bag, positive_argument_sentence_word_bag_padding = dataProcess_argument_train.get_sentence_word_bag( positive_argument_sentence_bag, Params.sentence_length, trigger_type_list)
    negative_argument_sentence_word_bag, negative_argument_sentence_word_bag_padding = dataProcess_argument_train.get_sentence_word_bag( negative_argument_sentence_bag, Params.sentence_length, trigger_type_list)
    positive_argument_word_position_list_bag, positive_argument_sentence_pre_word_list_bag = dataProcess_argument_train.get_sentence_word_position_and_sentence_pre_word( positive_argument_sentence_word_bag, positive_argument_sentence_word_bag_padding, mid_vocabulary, positive_argument_mid_pre_bag)
    negative_argument_word_position_list_bag, negative_argument_sentence_pre_word_list_bag = dataProcess_argument_train.get_sentence_word_position_and_sentence_pre_word( negative_argument_sentence_word_bag, negative_argument_sentence_word_bag_padding, mid_vocabulary, negative_argument_mid_pre_bag)
    positive_argument_sentence_pre_word_id_list_bag = dataProcess_argument_train.get_sentence_words_id_bag(vocabulary_id_dict, positive_argument_sentence_pre_word_list_bag)
    negative_argument_sentence_pre_word_id_list_bag = dataProcess_argument_train.get_sentence_words_id_bag(vocabulary_id_dict, negative_argument_sentence_pre_word_list_bag)
    positive_argument_word_adjacent_words_bag = dataProcess_argument_train.get_adjacent_words_bag(positive_argument_sentence_word_bag, mid_vocabulary, positive_argument_mid_pre_bag)
    negative_argument_word_adjacent_words_bag = dataProcess_argument_train.get_adjacent_words_bag(negative_argument_sentence_word_bag, mid_vocabulary, negative_argument_mid_pre_bag)
    positive_argument_adjacent_words_id_bag = dataProcess_argument_train.get_adjacent_words_id_bag(vocabulary_id_dict, positive_argument_word_adjacent_words_bag)
    negative_argument_adjacent_words_id_bag = dataProcess_argument_train.get_adjacent_words_id_bag(vocabulary_id_dict, negative_argument_word_adjacent_words_bag)
    positive_argument_trigger_adjacent_words_bag, positive_trigger_position_list_bag = dataProcess_argument_train.get_trigger_adjacent_words_bag_and_trigger_position_bag( positive_argument_sentence_word_bag, positive_argument_sentence_word_bag_padding, positive_argument_trigger_bag, mid_vocabulary, positive_argument_mid_pre_bag)
    negative_argument_trigger_adjacent_words_bag, negative_trigger_position_list_bag = dataProcess_argument_train.get_trigger_adjacent_words_bag_and_trigger_position_bag( negative_argument_sentence_word_bag, negative_argument_sentence_word_bag_padding, negative_argument_trigger_bag, mid_vocabulary, negative_argument_mid_pre_bag)
    positive_argument_event_type_bag, positive_argument_event_type_vector_pre_bag = dataProcess_argument_train.get_event_type_bag( positive_argument_sentence_word_bag, positive_event_type_bag, trigger_type_list, mid_vocabulary, positive_argument_mid_pre_bag)
    negative_argument_event_type_bag, negative_argument_event_type_vector_pre_bag = dataProcess_argument_train.get_event_type_bag(negative_argument_sentence_word_bag, negative_event_type_bag, trigger_type_list, mid_vocabulary, negative_argument_mid_pre_bag)
    positive_argument_trigger_adjacent_words_id_bag = dataProcess_argument_train.get_adjacent_words_id_bag(vocabulary_id_dict, positive_argument_trigger_adjacent_words_bag)
    negative_argument_trigger_adjacent_words_id_bag = dataProcess_argument_train.get_adjacent_words_id_bag(vocabulary_id_dict, negative_argument_trigger_adjacent_words_bag)
    positive_argument_trigger_parts_index_bag = dataProcess_argument_train.get_argument_parts_index_bag(positive_argument_word_position_list_bag, positive_trigger_position_list_bag, Params.sentence_length)
    negative_argument_trigger_parts_index_bag = dataProcess_argument_train.get_argument_parts_index_bag(negative_argument_word_position_list_bag, negative_trigger_position_list_bag, Params.sentence_length)

    positive_argument_bag_flags = []
    negative_argument_bag_flags = []

    for i in range(len(positive_argument_sentence_word_bag)):
        positive_argument_bag_flags.append(int(1))
    for i in range(len(negative_argument_sentence_word_bag)):
        negative_argument_bag_flags.append(int(0))

    word_list_bag = []
    sentence_pre_word_id_list_bag = []
    word_position_list_bag = []
    trigger_position_list_bag = []
    label_bag = []
    event_type_bag = []
    adjacent_words_id_bag = []
    trigger_adjacent_words_id_bag = []
    trigger_parts_index_bag = []
    bag_flags = []

    word_list_bag.extend(positive_argument_sentence_word_bag)
    word_list_bag.extend(negative_argument_sentence_word_bag)
    sentence_pre_word_id_list_bag.extend(positive_argument_sentence_pre_word_id_list_bag)
    sentence_pre_word_id_list_bag.extend(negative_argument_sentence_pre_word_id_list_bag)
    word_position_list_bag.extend(positive_argument_word_position_list_bag)
    word_position_list_bag.extend(negative_argument_word_position_list_bag)
    trigger_position_list_bag.extend(positive_trigger_position_list_bag)
    trigger_position_list_bag.extend(negative_trigger_position_list_bag)
    label_bag.extend(positive_sentence_bag_argument_type_vector)
    label_bag.extend(negative_sentence_bag_argument_type_vector)
    event_type_bag.extend(positive_argument_event_type_vector_pre_bag)
    event_type_bag.extend(negative_argument_event_type_vector_pre_bag)
    adjacent_words_id_bag.extend(positive_argument_adjacent_words_id_bag)
    adjacent_words_id_bag.extend(negative_argument_adjacent_words_id_bag)
    trigger_adjacent_words_id_bag.extend(positive_argument_trigger_adjacent_words_id_bag)
    trigger_adjacent_words_id_bag.extend(negative_argument_trigger_adjacent_words_id_bag)
    trigger_parts_index_bag.extend(positive_argument_trigger_parts_index_bag)
    trigger_parts_index_bag.extend(negative_argument_trigger_parts_index_bag)
    bag_flags.extend(positive_argument_bag_flags)
    bag_flags.extend(negative_argument_bag_flags)

    return word_list_bag, sentence_pre_word_id_list_bag, word_position_list_bag, trigger_position_list_bag, label_bag, event_type_bag, adjacent_words_id_bag, trigger_adjacent_words_id_bag, trigger_parts_index_bag, bag_flags


def get_argument_test_bag_tensor(sentence_filename, predict_info_path):
    """
    Get argument bag sample.
    :param sentence_filename:
    :return: Argument bag sample.
    """
    positive_bag_pre_labels, negative_bag_pre_labels, positive_bag_pre_trigger, negative_bag_pre_trigger, positive_bag_org_label, negative_bag_org_label = dataProacess_argument_val.extract_prediction_info(
        predict_info_path)
    vocabulary_id_dict = dataProacess_argument_val.read_vocabulary(Params.argument_vocabulary_path)
    mid_vocabulary = dataProacess_argument_val.read_mid_vocabulary(Params.mid_vocabulary_path)

    trigger_list, key_argument_list, argument_type_list, trigger_type_list = dataProacess_argument_val.get_trigger_and_key_argument(
        sentence_filename, Params.argument_type_path, Params.trigger_type_path)

    positive_argument_sentence_bag, negative_argument_sentence_bag, positive_sentence_bag_argument_type_vector, \
    negative_sentence_bag_argument_type_vector, positive_argument_trigger_bag, negative_argument_trigger_bag, \
    positive_event_type_bag, negative_event_type_bag, positive_trigger_bag_pre_labels_argument_bag, \
    negative_trigger_bag_pre_labels_argument_bag, positive_trigger_bag_org_labels_argument_bag, \
    negative_trigger_bag_org_labels_argument_bag, positive_argument_mid_pre_bag, negative_argument_mid_pre_bag = dataProacess_argument_val.get_sentence_argument_bag(
        sentence_filename, argument_type_list, key_argument_list,
        positive_bag_pre_labels, negative_bag_pre_labels, positive_bag_pre_trigger, negative_bag_pre_trigger,
        positive_bag_org_label, negative_bag_org_label)
    positive_argument_sentence_word_bag, positive_argument_sentence_word_bag_padding = dataProacess_argument_val.get_sentence_word_bag(
        positive_argument_sentence_bag, Params.sentence_length, trigger_type_list)
    negative_argument_sentence_word_bag, negative_argument_sentence_word_bag_padding = dataProacess_argument_val.get_sentence_word_bag(
        negative_argument_sentence_bag, Params.sentence_length, trigger_type_list)
    positive_argument_word_position_list_bag, positive_argument_sentence_pre_word_list_bag = dataProacess_argument_val.get_sentence_word_position_and_sentence_pre_word(
        positive_argument_sentence_word_bag, positive_argument_sentence_word_bag_padding, mid_vocabulary,
        positive_argument_mid_pre_bag)
    negative_argument_word_position_list_bag, negative_argument_sentence_pre_word_list_bag = dataProacess_argument_val.get_sentence_word_position_and_sentence_pre_word(
        negative_argument_sentence_word_bag, negative_argument_sentence_word_bag_padding, mid_vocabulary,
        negative_argument_mid_pre_bag)
    positive_argument_sentence_pre_word_id_list_bag = dataProacess_argument_val.get_sentence_words_id_bag(vocabulary_id_dict,
                                                                                positive_argument_sentence_pre_word_list_bag)
    negative_argument_sentence_pre_word_id_list_bag = dataProacess_argument_val.get_sentence_words_id_bag(vocabulary_id_dict,
                                                                                negative_argument_sentence_pre_word_list_bag)
    positive_argument_word_adjacent_words_bag = dataProacess_argument_val.get_adjacent_words_bag(positive_argument_sentence_word_bag,
                                                                       mid_vocabulary, positive_argument_mid_pre_bag)
    negative_argument_word_adjacent_words_bag = dataProacess_argument_val.get_adjacent_words_bag(negative_argument_sentence_word_bag,
                                                                       mid_vocabulary, negative_argument_mid_pre_bag)
    positive_argument_adjacent_words_id_bag = dataProacess_argument_val.get_adjacent_words_id_bag(vocabulary_id_dict,
                                                                        positive_argument_word_adjacent_words_bag)
    negative_argument_adjacent_words_id_bag = dataProacess_argument_val.get_adjacent_words_id_bag(vocabulary_id_dict,
                                                                        negative_argument_word_adjacent_words_bag)
    positive_argument_trigger_adjacent_words_bag, positive_trigger_position_list_bag = dataProacess_argument_val.get_trigger_adjacent_words_bag_and_trigger_position_bag(
        positive_argument_sentence_word_bag, positive_argument_sentence_word_bag_padding, positive_argument_trigger_bag,
        mid_vocabulary, positive_argument_mid_pre_bag)
    negative_argument_trigger_adjacent_words_bag, negative_trigger_position_list_bag = dataProacess_argument_val.get_trigger_adjacent_words_bag_and_trigger_position_bag(
        negative_argument_sentence_word_bag, negative_argument_sentence_word_bag_padding, negative_argument_trigger_bag,
        mid_vocabulary, negative_argument_mid_pre_bag)
    positive_argument_event_type_bag, positive_argument_event_type_vector_pre_bag = dataProacess_argument_val.get_event_type_bag(
        positive_argument_sentence_word_bag, positive_event_type_bag, trigger_type_list, mid_vocabulary,
        positive_argument_mid_pre_bag)
    negative_argument_event_type_bag, negative_argument_event_type_vector_pre_bag = dataProacess_argument_val.get_event_type_bag(
        negative_argument_sentence_word_bag, negative_event_type_bag, trigger_type_list, mid_vocabulary,
        negative_argument_mid_pre_bag)
    positive_argument_trigger_adjacent_words_id_bag = dataProacess_argument_val.get_adjacent_words_id_bag(vocabulary_id_dict,
                                                                                positive_argument_trigger_adjacent_words_bag)
    negative_argument_trigger_adjacent_words_id_bag = dataProacess_argument_val.get_adjacent_words_id_bag(vocabulary_id_dict,
                                                                                negative_argument_trigger_adjacent_words_bag)
    positive_argument_trigger_parts_index_bag = dataProacess_argument_val.get_argument_parts_index_bag(positive_argument_word_position_list_bag,
                                                                             positive_trigger_position_list_bag,
                                                                             Params.sentence_length)
    negative_argument_trigger_parts_index_bag = dataProacess_argument_val.get_argument_parts_index_bag(negative_argument_word_position_list_bag,
                                                                             negative_trigger_position_list_bag,
                                                                             Params.sentence_length)

    positive_argument_bag_flags = []
    negative_argument_bag_flags = []

    for i in range(len(positive_argument_sentence_word_bag)):
        positive_argument_bag_flags.append(int(1))
    for i in range(len(negative_argument_sentence_word_bag)):
        negative_argument_bag_flags.append(int(0))

    word_list_bag = []
    sentence_pre_word_id_list_bag = []
    word_position_list_bag = []
    trigger_position_list_bag = []
    label_bag = []
    event_type_bag = []
    adjacent_words_id_bag = []
    trigger_adjacent_words_id_bag = []
    trigger_parts_index_bag = []
    bag_flags = []

    word_list_bag.extend(positive_argument_sentence_word_bag)
    word_list_bag.extend(negative_argument_sentence_word_bag)
    sentence_pre_word_id_list_bag.extend(positive_argument_sentence_pre_word_id_list_bag)
    sentence_pre_word_id_list_bag.extend(negative_argument_sentence_pre_word_id_list_bag)
    word_position_list_bag.extend(positive_argument_word_position_list_bag)
    word_position_list_bag.extend(negative_argument_word_position_list_bag)
    trigger_position_list_bag.extend(positive_trigger_position_list_bag)
    trigger_position_list_bag.extend(negative_trigger_position_list_bag)
    label_bag.extend(positive_sentence_bag_argument_type_vector)
    label_bag.extend(negative_sentence_bag_argument_type_vector)
    event_type_bag.extend(positive_argument_event_type_vector_pre_bag)
    event_type_bag.extend(negative_argument_event_type_vector_pre_bag)
    adjacent_words_id_bag.extend(positive_argument_adjacent_words_id_bag)
    adjacent_words_id_bag.extend(negative_argument_adjacent_words_id_bag)
    trigger_adjacent_words_id_bag.extend(positive_argument_trigger_adjacent_words_id_bag)
    trigger_adjacent_words_id_bag.extend(negative_argument_trigger_adjacent_words_id_bag)
    trigger_parts_index_bag.extend(positive_argument_trigger_parts_index_bag)
    trigger_parts_index_bag.extend(negative_argument_trigger_parts_index_bag)
    bag_flags.extend(positive_argument_bag_flags)
    bag_flags.extend(negative_argument_bag_flags)

    return word_list_bag, sentence_pre_word_id_list_bag, word_position_list_bag, trigger_position_list_bag, label_bag, event_type_bag, adjacent_words_id_bag, trigger_adjacent_words_id_bag, trigger_parts_index_bag, bag_flags


def get_argument_bag_pre_tensor(sentence_filename, predict_info_path):
    """
    Get argument bag for info.
    :param sentence_filename:
    :return:
    """
    positive_bag_pre_labels, negative_bag_pre_labels, positive_bag_pre_trigger, negative_bag_pre_trigger, positive_bag_org_label, negative_bag_org_label = dataProacess_argument_val.extract_prediction_info(predict_info_path)
    vocabulary_id_dict = dataProacess_argument_val.read_vocabulary(Params.argument_vocabulary_path)
    mid_vocabulary = dataProacess_argument_val.read_mid_vocabulary(Params.mid_vocabulary_path)

    trigger_list, key_argument_list, argument_type_list, trigger_type_list = dataProacess_argument_val.get_trigger_and_key_argument(sentence_filename, Params.argument_type_path, Params.trigger_type_path)

    positive_argument_sentence_bag, negative_argument_sentence_bag, positive_sentence_bag_argument_type_vector, \
    negative_sentence_bag_argument_type_vector, positive_argument_trigger_bag, negative_argument_trigger_bag,  \
    positive_event_type_bag, negative_event_type_bag, positive_trigger_bag_pre_labels_argument_bag, \
    negative_trigger_bag_pre_labels_argument_bag, positive_trigger_bag_org_labels_argument_bag, \
    negative_trigger_bag_org_labels_argument_bag, positive_argument_mid_pre_bag, negative_argument_mid_pre_bag = dataProacess_argument_val.get_sentence_argument_bag(sentence_filename, argument_type_list, key_argument_list, positive_bag_pre_labels, negative_bag_pre_labels, positive_bag_pre_trigger, negative_bag_pre_trigger, positive_bag_org_label, negative_bag_org_label)

    # print(positive_trigger_bag_pre_labels_argument_bag)
    # print(positive_trigger_bag_org_labels_argument_bag)

    trigger_bag_pre_labels_argument_bag = []
    trigger_bag_org_labels_argument_bag = []
    trigger_bag_pre_labels_argument_bag.extend(positive_trigger_bag_pre_labels_argument_bag)
    trigger_bag_pre_labels_argument_bag.extend(negative_trigger_bag_pre_labels_argument_bag)
    trigger_bag_org_labels_argument_bag.extend(positive_trigger_bag_org_labels_argument_bag)
    trigger_bag_org_labels_argument_bag.extend(negative_trigger_bag_org_labels_argument_bag)

    return trigger_bag_pre_labels_argument_bag, trigger_bag_org_labels_argument_bag


if __name__ == "__main__":
    # word_list_bag, sentence_pre_word_id_list_bag, word_position_list_bag, trigger_position_list_bag, label_bag, \
    # event_type_bag, adjacent_words_id_bag, trigger_adjacent_words_id_bag, trigger_parts_index_bag, bag_flags = get_argument_bag_tensor("./data/bag_temp.txt")

    trigger_bag_pre_labels_argument_bag, trigger_bag_org_labels_argument_bag = get_argument_bag_pre_tensor(Params.bag_test_wiki_sentence_annotated_with_trigger_path, Params.bag_test_predict_info_path)
    # print(trigger_bag_pre_labels_argument_bag)
    # print(trigger_bag_org_labels_argument_bag)
    # print(word_list_bag)
    # print(sentence_pre_word_id_list_bag)
    # print(word_position_list_bag)
    # print(trigger_position_list_bag)
    # print(label_bag)
    # print(event_type_bag)
    # print(adjacent_words_id_bag)
    # print(trigger_adjacent_words_id_bag)
    # print(trigger_parts_index_bag)
    # print(bag_flags)