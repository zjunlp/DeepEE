# -*- coding: utf-8 -*-

"""
Tagger for argument.
Author:ZXY
Date:12/19/2017
"""

import os
import Params
from DMCNN_argument import DMCNN_ARGUMENT
import tensor_argument
import tensorflow as tf
import numpy as np
import random
import time
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# tf_config = tf.ConfigProto(allow_soft_placement=True,
#                            log_device_placement=True)
# tf_config.gpu_options.allow_growth = True

# Parameters
# =========================================================================

tf.flags.DEFINE_boolean("train_param", True, "Train or not")

# Data load parameters
tf.flags.DEFINE_string("positive_sample_path", Params.positive_sample_path, "Positive sample file")
tf.flags.DEFINE_string("negative_sample_path", Params.negative_sample_path, "Negative sample file")

# Model parameters
tf.flags.DEFINE_integer("sentence_length", Params.sentence_length, "The length of sentence")
tf.flags.DEFINE_integer("num_role", Params.argument_num_role, "The number of role")
tf.flags.DEFINE_integer("event_type_num_role", Params.event_type_num_role, "The number of role")
tf.flags.DEFINE_integer("vocabulary_size", Params.argument_vocabulary_size, "The size of vocabulary")
tf.flags.DEFINE_integer("embedding_size", Params.embedding_size, "The size of embedding")
tf.flags.DEFINE_integer("position_size", Params.position_size, "The size of word position")
tf.flags.DEFINE_integer("position_embedding_size", Params.position_embedding_size,
                        "The size of word position embedding")
tf.flags.DEFINE_integer("event_type_embedding_size", Params.event_type_embedding_size,
                        "The size of event type embedding")
tf.flags.DEFINE_integer("filter_size", Params.filter_size, "The size of filter")
tf.flags.DEFINE_integer("num_filters", Params.num_filters, "the number of filter")
tf.flags.DEFINE_integer("num_epochs", Params.num_epochs, "The number of epoch")
tf.flags.DEFINE_integer("bag_word_length", Params.bag_argument_word_length, "The length of bag word")

# Training parameters
tf.flags.DEFINE_integer("batch_size", Params.bag_batch_size, "The size of batch")
tf.flags.DEFINE_float("dropout_keep_prob", Params.dropout_keep_prob_test, "The dropout of training")
tf.flags.DEFINE_integer("checkpoint_pre", Params.checkpoint_pre, "The number of checkpoint pre")
tf.flags.DEFINE_integer("evaluate_pre", Params.evaluate_pre, "The number of evaluate pre")
tf.flags.DEFINE_integer("cross_fold", Params.cross_fold, "The number of cross_fold")
tf.flags.DEFINE_integer("save_pre", Params.save_pre, "The number of save pre")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data preparation
# ====================================================================
print(str(Params.cross_fold))

dev_word_list_bag, dev_sentence_pre_word_id_list_bag, dev_word_position_list_bag, dev_trigger_position_list_bag, \
dev_label_bag, dev_event_type_bag, dev_adjacent_words_id_bag, dev_trigger_adjacent_words_id_bag, \
dev_trigger_parts_index_bag, dev_bag_flags = tensor_argument.get_argument_test_bag_tensor(Params.bag_dev_wiki_sentence_annotated_with_trigger_path, Params.bag_dev_predict_info_path_1)

# print(train_trigger_adjacent_words_id_bag)

trigger_bag_pre_labels_argument_bag, trigger_bag_org_labels_argument_bag = tensor_argument.get_argument_bag_pre_tensor(Params.bag_dev_wiki_sentence_annotated_with_trigger_path, Params.bag_dev_predict_info_path_1)

dev_combine_sample = list(zip(dev_word_list_bag, dev_sentence_pre_word_id_list_bag, dev_word_position_list_bag, dev_trigger_position_list_bag,
                              dev_label_bag, dev_event_type_bag, dev_adjacent_words_id_bag, dev_trigger_adjacent_words_id_bag,
                              dev_trigger_parts_index_bag, dev_bag_flags, trigger_bag_pre_labels_argument_bag, trigger_bag_org_labels_argument_bag))
# print(len(dev_combine_sample))
# random.shuffle(dev_combine_sample)
# print(len(dev_combine_sample))


def tagger_argument():
    # Tagger
    # ====================================================================
    with tf.Graph().as_default(), tf.device('/gpu:1'):
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement
        )
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            DMcnn_argument = DMCNN_ARGUMENT(
                sequence_length=FLAGS.sentence_length,
                num_role=FLAGS.num_role,
                vocab_size=FLAGS.vocabulary_size,
                embedding_size=FLAGS.embedding_size,
                position_size=FLAGS.position_size,
                position_embedding_size=FLAGS.position_embedding_size,
                event_type_num=FLAGS.num_role,
                event_type_embedding_size=FLAGS.event_type_embedding_size,
                filter_size=FLAGS.filter_size,
                num_filters=FLAGS.num_filters,
                trigger_flags=1,
                gpu_flags='/gpu:1',
                bag_word_length=FLAGS.bag_word_length
            )

            saver = tf.train.Saver(tf.global_variables())
            # saver.restore(sess, "./argument/1514209540/checkpoints/model-110000")
            # saver.restore(sess, "./argument/1514209540/checkpoints/model-183000")
            # saver.restore(sess, "./argument/1514209540/checkpoints/model-172500")

            saver.restore(sess, "./argument/1514209540/checkpoints/model-188000")

            # Load data
            dev_word_list_bag, dev_sentence_pre_word_id_list_bag, dev_word_position_list_bag, dev_trigger_position_list_bag, \
            dev_label_bag, dev_event_type_bag, dev_adjacent_words_id_bag, dev_trigger_adjacent_words_id_bag, \
            dev_trigger_parts_index_bag, dev_bag_flags, dev_trigger_bag_pre_labels_argument_bag, dev_trigger_bag_org_labels_argument_bag = split_sample(dev_combine_sample)

            # Dev
            # ==========================================
            batch_dev_num = 0
            loss_dev = 0.0
            acc_dev = 0.0
            pos_pre_all_dev = 0
            pos_pre_num_dev = 0
            pos_all_num_dev = 0
            # ============================================
            batches_dev = batch_iter(list(zip(dev_sentence_pre_word_id_list_bag, dev_word_position_list_bag,
                                                dev_trigger_position_list_bag, dev_label_bag, dev_event_type_bag,
                                                dev_adjacent_words_id_bag, dev_trigger_adjacent_words_id_bag,
                                                dev_trigger_parts_index_bag, dev_bag_flags, dev_trigger_bag_pre_labels_argument_bag,
                                              dev_trigger_bag_org_labels_argument_bag)), FLAGS.batch_size, 0)

            for batch_dev in batches_dev:
                batch_dev_num += 1

                sentence_pre_word_id_list_bag_dev_batch, word_position_list_bag_dev_batch, \
                trigger_position_list_bag_dev_batch, label_bag_dev_batch, event_type_bag_dev_batch, \
                adjacent_words_id_bag_dev_batch, trigger_adjacent_words_id_bag_dev_batch, \
                trigger_parts_index_bag_dev_batch, bag_flags_dev_batch, trigger_bag_pre_labels_argument_bag_dev_batch, \
                dev_trigger_bag_org_labels_argument_bag_dev_batch = zip(*batch_dev)
                bag_num_dev_batch = len(label_bag_dev_batch)

                sample_label_neg_batch_dev = []
                for i in range(len(label_bag_dev_batch)):
                    sample_neg_flags_temp = [int(0)] * 61
                    sample_neg_flags_temp.append(int(1))
                    sample_label_neg_batch_dev.append(sample_neg_flags_temp)
                sample_label_neg_batch_dev = np.array(sample_label_neg_batch_dev)

                scores_lookup_add_dev = []
                scores_lookup_add_dev.append(int(0))
                id_dev_num = 0
                for i in range(len(label_bag_dev_batch) - 1):
                    id_dev_num += Params.bag_argument_word_length
                    scores_lookup_add_dev.append(int(id_dev_num))
                scores_lookup_add_dev = np.array(scores_lookup_add_dev)

                sentence_pre_word_id_list_bag_all_dev_batch = []
                word_position_list_bag_all_dev_batch = []
                trigger_position_list_bag_all_dev_batch = []
                label_bag_all_dev_batch = []
                event_type_bag_all_dev_batch = []
                adjacent_words_id_bag_all_dev_batch = []
                trigger_adjacent_words_id_bag_all_dev_batch = []
                trigger_parts_index_bag_all_dev_batch = []
                bag_flags_all_dev_batch = []

                for bag_num in range(len(label_bag_dev_batch)):
                    sentence_pre_word_id_list_bag_dev_all = []
                    word_position_list_bag_dev_all = []
                    trigger_position_list_bag_dev_all = []
                    # label_bag_dev_all = []
                    event_type_bag_dev_all = []
                    adjacent_words_id_bag_dev_all = []
                    trigger_adjacent_words_id_bag_dev_all = []
                    trigger_parts_index_bag_dev_all = []
                    bag_flags_dev_all = []

                    for sentence_num in range(len(sentence_pre_word_id_list_bag_dev_batch[bag_num])):
                        sentence_pre_word_id_list_bag_dev_all.extend(sentence_pre_word_id_list_bag_dev_batch[bag_num][sentence_num])
                        word_position_list_bag_dev_all.extend(word_position_list_bag_dev_batch[bag_num][sentence_num])
                        trigger_position_list_bag_dev_all.extend(trigger_position_list_bag_dev_batch[bag_num][sentence_num])
                        event_type_bag_dev_all.extend(event_type_bag_dev_batch[bag_num][sentence_num])
                        adjacent_words_id_bag_dev_all.extend(adjacent_words_id_bag_dev_batch[bag_num][sentence_num])
                        trigger_adjacent_words_id_bag_dev_all.extend(trigger_adjacent_words_id_bag_dev_batch[bag_num][sentence_num])
                        trigger_parts_index_bag_dev_all.extend(trigger_parts_index_bag_dev_batch[bag_num][sentence_num])
                        # word_num = int(len(sentence_pre_word_id_list_bag_dev_all))

                    sentence_pre_word_id_list_bag_dev_all, word_position_list_bag_dev_all, trigger_position_list_bag_dev_all, \
                    event_type_bag_dev_all, adjacent_words_id_bag_dev_all, trigger_adjacent_words_id_bag_dev_all, trigger_parts_index_bag_dev_all \
                        = cut_bag_word(sentence_pre_word_id_list_bag_dev_all, word_position_list_bag_dev_all,
                                        trigger_position_list_bag_dev_all, event_type_bag_dev_all,adjacent_words_id_bag_dev_all,
                                        trigger_adjacent_words_id_bag_dev_all, trigger_parts_index_bag_dev_all,
                                        Params.bag_argument_word_length)

                    # sentence_pre_word_id_list_bag_dev_all = np.array(sentence_pre_word_id_list_bag_dev_all)
                    # word_position_list_bag_dev_all = np.array(word_position_list_bag_dev_all)
                    # label_bag_dev_all.append(label_bag_dev_batch[bag_num])
                    # label_bag_dev_all = np.array(label_bag_dev_all)
                    label_bag_dev_all = np.array(label_bag_dev_batch[bag_num])
                    bag_flags_dev_all.append(bag_flags_dev_batch[bag_num])
                    # bag_flags_dev_all = np.array(bag_flags_dev_all)
                    # adjacent_words_id_bag_dev_all = np.array(adjacent_words_id_bag_dev_all)
                    # trigger_parts_index_bag_dev_all = np.array(trigger_parts_index_bag_dev_all)

                    sentence_pre_word_id_list_bag_all_dev_batch.extend(sentence_pre_word_id_list_bag_dev_all)
                    word_position_list_bag_all_dev_batch.extend(word_position_list_bag_dev_all)
                    trigger_position_list_bag_all_dev_batch.extend(trigger_position_list_bag_dev_all)
                    label_bag_all_dev_batch.append(label_bag_dev_all)
                    event_type_bag_all_dev_batch.extend(event_type_bag_dev_all)
                    adjacent_words_id_bag_all_dev_batch.extend(adjacent_words_id_bag_dev_all)
                    trigger_adjacent_words_id_bag_all_dev_batch.extend(trigger_adjacent_words_id_bag_dev_all)
                    trigger_parts_index_bag_all_dev_batch.extend(trigger_parts_index_bag_dev_all)
                    bag_flags_all_dev_batch.extend(bag_flags_dev_all)

                sentence_pre_word_id_list_bag_all_dev_batch = np.array(sentence_pre_word_id_list_bag_all_dev_batch)
                word_position_list_bag_all_dev_batch = np.array(word_position_list_bag_all_dev_batch)
                trigger_position_list_bag_all_dev_batch = np.array(trigger_position_list_bag_all_dev_batch)
                event_type_bag_all_dev_batch = np.array(event_type_bag_all_dev_batch)
                label_bag_all_dev_batch = np.array(label_bag_all_dev_batch)
                adjacent_words_id_bag_all_dev_batch = np.array(adjacent_words_id_bag_all_dev_batch)
                trigger_adjacent_words_id_bag_all_dev_batch = np.array(trigger_adjacent_words_id_bag_all_dev_batch)
                trigger_parts_index_bag_all_dev_batch = np.array(trigger_parts_index_bag_all_dev_batch)
                bag_flags_all_dev_batch = np.array(bag_flags_all_dev_batch)

                feed_dict = {DMcnn_argument.input_sentence: sentence_pre_word_id_list_bag_all_dev_batch,
                                DMcnn_argument.input_word_position: word_position_list_bag_all_dev_batch,
                                DMcnn_argument.input_trigger_position: trigger_position_list_bag_all_dev_batch,
                                DMcnn_argument.input_type: event_type_bag_all_dev_batch,
                                DMcnn_argument.input_role: label_bag_all_dev_batch,
                                DMcnn_argument.input_parts_indexs: trigger_parts_index_bag_all_dev_batch,
                                DMcnn_argument.input_adjacent_words: adjacent_words_id_bag_all_dev_batch,
                                DMcnn_argument.input_trigger_adjacent_words: trigger_adjacent_words_id_bag_all_dev_batch,
                                DMcnn_argument.bag_num: bag_num_dev_batch,
                                DMcnn_argument.input_flags: bag_flags_all_dev_batch,
                                DMcnn_argument.input_neg_label: sample_label_neg_batch_dev,
                                DMcnn_argument.scores_lookup_add: scores_lookup_add_dev,
                                DMcnn_argument.dropout_keep_prob: FLAGS.dropout_keep_prob
                                }

                loss, accuracy, pos_pre, pos_pre_all, pos_pre_num, pos_all_num, precision, recall, F1, scores_max, scores_max_id, prediction_labels, input_labels, scores_flat = sess.run(
                    [DMcnn_argument.loss, DMcnn_argument.accuracy,
                        DMcnn_argument.positive_correct_prediction,
                        DMcnn_argument.positive_prediction,
                        DMcnn_argument.positive_correct_prediction_num,
                        DMcnn_argument.positive_correct_all_num,
                        DMcnn_argument.precision,
                        DMcnn_argument.recall_value,
                        DMcnn_argument.F1_value,
                        DMcnn_argument.scores_max_score,
                        DMcnn_argument.scores_lookup_ids,
                        DMcnn_argument.prediction,
                        DMcnn_argument.org_label,
                        DMcnn_argument.scores_flat], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                loss_dev = loss_dev + loss
                acc_dev = acc_dev + accuracy
                pos_pre_all_dev = pos_pre_all_dev + pos_pre_all
                pos_pre_num_dev = pos_pre_num_dev + pos_pre_num
                pos_all_num_dev = pos_all_num_dev + pos_all_num

                # ================================Pre info==============================================
                scores_max_ids = []
                scores_maxs = []
                org_labels = []
                pre_labels = []
                for i in range(len(list(bag_flags_dev_batch))):
                    scores_max_ids.append(int(scores_max_id[i]))
                    scores_maxs.append(float(scores_max[i]))
                    org_labels.append(input_labels[i])
                    pre_labels.append(prediction_labels[i])
                write_bag_predict_info(scores_maxs, scores_max_ids, pre_labels, org_labels, trigger_bag_pre_labels_argument_bag_dev_batch,
                                       dev_trigger_bag_org_labels_argument_bag_dev_batch, Params.bag_argument_withou_trigger_dev_predict_info_path)

                print("Batch done!" + str(batch_dev_num))
            loss = loss_dev / batch_dev_num
            accuracy = acc_dev / batch_dev_num
            precision = pos_pre_num_dev / pos_pre_all_dev
            recall = pos_pre_num_dev / pos_all_num_dev
            F1 = (2 * precision * recall) / (precision + recall)

            print("Dev: {}: loss {:g}, acc {:g}, pre {:g}, recall {:g}, F1 {:g}".format(time_str, loss, accuracy, precision, recall, F1))
            print('\n')
            print(pos_pre_all_dev)
            print(pos_pre_num_dev)
            print(pos_all_num_dev)
            print('\n')


def cut_bag_word(sentence_pre_word_id_list_bag_all, word_position_list_bag_all, trigger_position_list_bag_all, event_type_bag_all, adjacent_words_id_bag_all, trigger_adjacent_words_id_bag_all, trigger_parts_index_bag_all, bag_argument_word_length):
    """
    Cut bag word
    :param sentence_pre_word_id_list_bag_all:
    :param word_position_list_bag_all:
    :param adjacent_words_id_bag_all:
    :param trigger_parts_index_bag_all:
    :return:
    """
    # print(trigger_adjacent_words_id_bag_all)
    sentence_pre_word_id_list_bag_all_cut = []
    word_position_list_bag_all_cut = []
    trigger_position_list_bag_all_cut = []
    event_type_bag_all_cut = []
    adjacent_words_id_bag_all_cut = []
    trigger_adjacent_words_id_bag_all_cut = []
    trigger_parts_index_bag_all_cut = []
    if len(sentence_pre_word_id_list_bag_all) <= bag_argument_word_length:
        if len(sentence_pre_word_id_list_bag_all) == 0:
            print(sentence_pre_word_id_list_bag_all)
            print(word_position_list_bag_all)
            print(trigger_position_list_bag_all)
            print(event_type_bag_all)
            print(adjacent_words_id_bag_all)
            print(trigger_adjacent_words_id_bag_all)
            print(trigger_parts_index_bag_all)
        num = bag_argument_word_length - len(sentence_pre_word_id_list_bag_all)

        sentence_pre_word_id_list_bag_all_cut = sentence_pre_word_id_list_bag_all
        word_position_list_bag_all_cut = word_position_list_bag_all
        trigger_position_list_bag_all_cut = trigger_position_list_bag_all
        event_type_bag_all_cut = event_type_bag_all
        adjacent_words_id_bag_all_cut = adjacent_words_id_bag_all
        trigger_adjacent_words_id_bag_all_cut = trigger_adjacent_words_id_bag_all
        trigger_parts_index_bag_all_cut = trigger_parts_index_bag_all

        for i in range(num):
            sentence_pre_word_id_list_bag_all_cut.append(sentence_pre_word_id_list_bag_all[0])
            word_position_list_bag_all_cut.append(word_position_list_bag_all[0])
            trigger_position_list_bag_all_cut.append(trigger_position_list_bag_all[0])
            event_type_bag_all_cut.append(event_type_bag_all[0])
            adjacent_words_id_bag_all_cut.append(adjacent_words_id_bag_all[0])
            trigger_adjacent_words_id_bag_all_cut.append(trigger_adjacent_words_id_bag_all[0])
            trigger_parts_index_bag_all_cut.append(trigger_parts_index_bag_all[0])
    else:
        for i in range(bag_argument_word_length):
            sentence_pre_word_id_list_bag_all_cut.append(sentence_pre_word_id_list_bag_all[i])
            word_position_list_bag_all_cut.append(word_position_list_bag_all[i])
            trigger_position_list_bag_all_cut.append(trigger_position_list_bag_all[i])
            event_type_bag_all_cut.append(event_type_bag_all[i])
            adjacent_words_id_bag_all_cut.append(adjacent_words_id_bag_all[i])
            trigger_adjacent_words_id_bag_all_cut.append(trigger_adjacent_words_id_bag_all[i])
            trigger_parts_index_bag_all_cut.append(trigger_parts_index_bag_all[i])
    # print(trigger_adjacent_words_id_bag_all_cut)

    return sentence_pre_word_id_list_bag_all_cut, word_position_list_bag_all_cut, trigger_position_list_bag_all_cut, event_type_bag_all_cut, adjacent_words_id_bag_all_cut, trigger_adjacent_words_id_bag_all_cut, trigger_parts_index_bag_all_cut


def split_sample(combine_sample):
    """
    Split Sample
    :param combine_sample:
    :return: split sample
    """
    word_list_bag, sentence_pre_word_id_list_bag, word_position_list_bag, trigger_position_list_bag, label_bag, \
    event_type_bag, adjacent_words_id_bag, trigger_adjacent_words_id_bag, trigger_parts_index_bag, bag_flags, trigger_bag_pre_labels_argument_bag, trigger_bag_org_labels_argument_bag = zip(*combine_sample)
    print(len(label_bag))
    #
    # # Train and test data spilt
    # # ==================================================================
    # sentence_pre_word_id_list_bag_dev = sentence_pre_word_id_list_bag[
    #                                     int((Round - 1) * train_sample_total / FLAGS.cross_fold): int(Round * train_sample_total / FLAGS.cross_fold)]
    # word_position_list_bag_dev = word_position_list_bag[
    #                              int((Round - 1) * train_sample_total / FLAGS.cross_fold): int(Round * train_sample_total / FLAGS.cross_fold)]
    # label_bag_dev = label_bag[int((Round - 1) * train_sample_total / FLAGS.cross_fold): int(Round * train_sample_total / FLAGS.cross_fold)]
    # trigger_parts_index_bag_dev = trigger_parts_index_bag[
    #                               int((Round - 1) * train_sample_total / FLAGS.cross_fold): int(Round * train_sample_total / FLAGS.cross_fold)]
    # adjacent_words_id_bag_dev = adjacent_words_id_bag[
    #                             int((Round - 1) * train_sample_total / FLAGS.cross_fold): int(Round * train_sample_total / FLAGS.cross_fold)]
    #
    # # sample_word_dev = np.array(sample_word_dev)
    # # sample_sentence_dev = np.array(sample_sentence_dev)
    # # sample_position_dev = np.array(sample_position_dev)
    # # label_dev = np.array(label_dev)
    # # smaple_adjacent_words_dev = np.array(smaple_adjacent_words_dev)
    # # sample_parts_indexs_dev = np.array(sample_parts_indexs_dev)
    # # sample_flags_dev = np.array(sample_flags_dev)
    # # print(sample_parts_indexs_dev)
    # # sample_parts_indexs_dev = tf.transpose(sample_parts_indexs_dev, [1, 2, 0])
    # # print(sample_parts_indexs_dev)
    #
    # sentence_pre_word_id_list_bag_train = sentence_pre_word_id_list_bag[int(Round * train_sample_total / FLAGS.cross_fold):] + \
    #                                       sentence_pre_word_id_list_bag[:int((Round - 1) * train_sample_total / FLAGS.cross_fold)]
    # word_position_list_bag_train = word_position_list_bag[int(Round * train_sample_total / FLAGS.cross_fold):] + \
    #                                word_position_list_bag[:int((Round - 1) * train_sample_total / FLAGS.cross_fold)]
    # label_bag_train = label_bag[int(Round * train_sample_total / FLAGS.cross_fold):] + \
    #                   label_bag[:int((Round - 1) * train_sample_total / FLAGS.cross_fold)]
    # trigger_parts_index_bag_train = trigger_parts_index_bag[int(Round * train_sample_total / FLAGS.cross_fold):] + \
    #                                 trigger_parts_index_bag[:int((Round - 1) * train_sample_total / FLAGS.cross_fold)]
    # # sample_parts_indexs_train = tf.transpose(sample_parts_indexs_train, [1, 2, 0])
    # adjacent_words_id_bag_train = adjacent_words_id_bag[int(Round * train_sample_total / FLAGS.cross_fold):] + \
    #                               adjacent_words_id_bag[:int((Round - 1) * train_sample_total / FLAGS.cross_fold)]
    #
    # print("Train/Dev split: {:d}/{:d}".format(len(label_bag_train), len(label_bag_dev)))

    return word_list_bag, sentence_pre_word_id_list_bag, word_position_list_bag, trigger_position_list_bag, label_bag, \
           event_type_bag, adjacent_words_id_bag, trigger_adjacent_words_id_bag, trigger_parts_index_bag, bag_flags, trigger_bag_pre_labels_argument_bag, trigger_bag_org_labels_argument_bag


def batch_iter(sample, batch_size, print_flags):
    """
    Batch create function
    :param sample:
    :param batch_size:
    :return: batch sample
    """
    # sample = np.array(sample)
    sample_size = len(sample)
    if sample_size % batch_size == 0:
        batches_num = int(sample_size/batch_size)
    else:
        batches_num = int(sample_size/batch_size) + 1
    for i in range(batches_num):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, sample_size)
        if print_flags == 1:
            print("Batch index %d" % end_index)
        yield sample[start_index: end_index]


def write_bag_predict_info(scores_maxs, scores_max_ids, pre_labels, org_labels, dev_trigger_bag_pre_labels_argument_bag, dev_trigger_bag_org_labels_argument_bag, output_filename):
    """
    Process bag predict information for sort P-R curve.
    :param scores_maxs:
    :param scores_max_ids:
    :param pre_labels:
    :param org_labels:
    :param dev_trigger_bag_pre_labels_argument_bag:
    :param dev_trigger_bag_org_labels_argument_bag:
    :param output_filename: Max scores word in vocabulary
    :return: Process and write into file.
    """
    print("Write result info into file...")
    with open(output_filename, "a", encoding="utf-8") as output_file:
        for i in range(len(scores_maxs)):
            output_file.write(str(scores_maxs[i]) + " ")
            output_file.write(str(scores_max_ids[i]) + " ")
            output_file.write(str(pre_labels[i]) + " ")
            output_file.write(str(org_labels[i]) + " ")
            output_file.write(str(dev_trigger_bag_pre_labels_argument_bag[i]) + " ")
            output_file.write(str(dev_trigger_bag_org_labels_argument_bag[i]) + " ")
            output_file.write("\n")
        output_file.close()


if __name__ == "__main__":
    tagger_argument()