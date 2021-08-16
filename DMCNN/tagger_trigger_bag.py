# -*- coding:utf-8 -*-
"""
Author:ZXY
Date: 09/14/2017
"""

from DMCNN_trigger_bag import DMCNN_TRIGGER_BAG
import Params
import tensor_trigger_bag
import random
import time
import datetime
import os
import tensorflow as tf
import numpy as np
import dataProcess_trigger_bag

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# tf_config = tf.ConfigProto(allow_soft_placement=True,
#                            log_device_placement=True)
# tf_config.gpu_options.allow_growth = True

# Parameters
# =========================================================================

# Data load parameters
tf.flags.DEFINE_string("positive_sample_path", Params.positive_sample_path, "Positive sample file")
tf.flags.DEFINE_string("negative_sample_path", Params.negative_sample_path, "Negative sample file")

# Model parameters
tf.flags.DEFINE_integer("sentence_length", Params.sentence_length, "The length of sentence")
tf.flags.DEFINE_integer("num_role", Params.num_role, "The number of role")
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
tf.flags.DEFINE_integer("bag_word_length", Params.bag_word_length, "The length of bag word")

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

dev_word_list_bag, dev_sentence_pre_word_id_list_bag, dev_word_position_list_bag, dev_label_bag, dev_adjacent_words_id_bag, dev_trigger_parts_index_bag, dev_bag_flags = \
    tensor_trigger_bag.get_trigger_bag_tensor(Params.bag_dev_wiki_sentence_annotated_with_trigger_path)
# print(dev_word_position_list_bag)
# print(dev_trigger_parts_index_bag)
dev_bag_org_ids, dev_pos_neg_bag_org_ids, dev_sentences_word_num = tensor_trigger_bag.get_trigger_bag_pre_tensor(Params.bag_dev_wiki_sentence_annotated_with_trigger_path)

dev_combine_sample = list(zip(dev_word_list_bag, dev_sentence_pre_word_id_list_bag, dev_word_position_list_bag, dev_label_bag, dev_adjacent_words_id_bag,
                              dev_trigger_parts_index_bag, dev_bag_flags, dev_bag_org_ids, dev_pos_neg_bag_org_ids, dev_sentences_word_num))
print(len(dev_combine_sample))
# random.shuffle(dev_combine_sample)
# print(len(dev_combine_sample))


def tagger_trigger_bag():
    with tf.Graph().as_default(), tf.device('/gpu:1'):
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement
        )
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            DMcnn_trigger_bag = DMCNN_TRIGGER_BAG(
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
                trigger_flags=0,
                gpu_flags='/gpu:1',
                bag_word_length=FLAGS.bag_word_length
            )

            saver = tf.train.Saver(tf.global_variables())
            # saver.restore(sess, "./trigger_bag/1513872441/checkpoints/model-124000")
            # main
            # saver.restore(sess, "./trigger_bag/1513872441/checkpoints/model-40000")

            # multi
            # saver.restore(sess, "./trigger_bag/1513872441/checkpoints/model-110000")

            # no
            saver.restore(sess, "./trigger_bag/1513872441/checkpoints/model-105500")
            # =======================Load Data=======================================================================
            vocabulary_id_dict = dataProcess_trigger_bag.read_vocabulary(Params.vocabulary_path)

            word_list_bag_dev, sentence_pre_word_id_list_bag_dev, word_position_list_bag_dev, label_bag_dev, \
            adjacent_words_id_bag_dev, trigger_parts_index_bag_dev, dev_bag_flags, dev_bag_org_ids, dev_pos_neg_bag_org_ids, dev_sentences_word_num = split_sample(dev_combine_sample)
            # ========================================================================================================
            batch_dev_num = 0
            loss_dev = 0.0
            acc_dev = 0.0
            pos_pre_all_dev = 0
            pos_pre_num_dev = 0
            pos_all_num_dev = 0
            # ======================================================================================
            batches_dev = batch_iter(
                list(zip(word_list_bag_dev, sentence_pre_word_id_list_bag_dev, word_position_list_bag_dev, label_bag_dev,
                         adjacent_words_id_bag_dev, trigger_parts_index_bag_dev, dev_bag_flags, dev_bag_org_ids, dev_pos_neg_bag_org_ids, dev_sentences_word_num)),
                FLAGS.batch_size, 0)
            for batch_dev in batches_dev:
                batch_dev_num += 1

                word_list_bag_dev_batch, sentence_pre_word_id_list_bag_dev_batch, word_position_list_bag_dev_batch, label_bag_dev_batch, \
                adjacent_words_id_bag_dev_batch, trigger_parts_index_bag_dev_batch, bag_flag_dev_batch, dev_bag_org_ids_batch, dev_pos_neg_bag_org_ids_batch, dev_sentences_word_num_batch = zip(*batch_dev)
                bag_num_dev_batch = len(label_bag_dev_batch)

                sample_label_neg_batch_dev = []
                for i in range(len(label_bag_dev_batch)):
                    sample_neg_flags_temp = [int(0)] * 21
                    sample_neg_flags_temp.append(int(1))
                    sample_label_neg_batch_dev.append(sample_neg_flags_temp)
                sample_label_neg_batch_dev = np.array(sample_label_neg_batch_dev)

                scores_lookup_add_dev = []
                scores_lookup_add_dev.append(int(0))
                id_dev_num = 0
                for i in range(len(label_bag_dev_batch) - 1):
                    id_dev_num += Params.bag_word_length
                    scores_lookup_add_dev.append(int(id_dev_num))
                scores_lookup_add_dev = np.array(scores_lookup_add_dev)

                word_bag_all_dev_batch = []
                sentence_pre_word_id_list_bag_all_dev_batch = []
                word_position_list_bag_all_dev_batch = []
                label_bag_all_dev_batch = []
                adjacent_words_id_bag_all_dev_batch = []
                trigger_parts_index_bag_all_dev_batch = []
                bag_flags_all_dev_batch = []

                for bag_num in range(len(label_bag_dev_batch)):
                    word_bag_dev_all = []
                    sentence_pre_word_id_list_bag_dev_all = []
                    word_position_list_bag_dev_all = []
                    # label_bag_dev_all = []
                    adjacent_words_id_bag_dev_all = []
                    trigger_parts_index_bag_dev_all = []
                    bag_flags_dev_all = []
                    for sentence_num in range(len(sentence_pre_word_id_list_bag_dev_batch[bag_num])):
                        word_bag_dev_all.extend(word_list_bag_dev_batch[bag_num][sentence_num])
                        sentence_pre_word_id_list_bag_dev_all.extend(
                            sentence_pre_word_id_list_bag_dev_batch[bag_num][sentence_num])
                        word_position_list_bag_dev_all.extend(word_position_list_bag_dev_batch[bag_num][sentence_num])
                        adjacent_words_id_bag_dev_all.extend(adjacent_words_id_bag_dev_batch[bag_num][sentence_num])
                        trigger_parts_index_bag_dev_all.extend(trigger_parts_index_bag_dev_batch[bag_num][sentence_num])
                        # word_num_dev = int(len(sentence_pre_word_id_list_bag_dev_all))

                    word_bag_dev_all, sentence_pre_word_id_list_bag_dev_all, word_position_list_bag_dev_all, adjacent_words_id_bag_dev_all, trigger_parts_index_bag_dev_all = \
                        cut_bag_word(word_bag_dev_all, sentence_pre_word_id_list_bag_dev_all, word_position_list_bag_dev_all,
                                     adjacent_words_id_bag_dev_all, trigger_parts_index_bag_dev_all,
                                     Params.bag_word_length)

                    label_bag_dev_all = np.array(label_bag_dev_batch[bag_num])
                    bag_flags_dev_all.append(bag_flag_dev_batch[bag_num])

                    word_bag_all_dev_batch.extend(word_bag_dev_all)
                    sentence_pre_word_id_list_bag_all_dev_batch.extend(sentence_pre_word_id_list_bag_dev_all)
                    word_position_list_bag_all_dev_batch.extend(word_position_list_bag_dev_all)
                    label_bag_all_dev_batch.append(label_bag_dev_all)
                    adjacent_words_id_bag_all_dev_batch.extend(adjacent_words_id_bag_dev_all)
                    trigger_parts_index_bag_all_dev_batch.extend(trigger_parts_index_bag_dev_all)
                    bag_flags_all_dev_batch.extend(bag_flags_dev_all)

                # word_bag_all_dev_batch = np.array(word_bag_all_dev_batch)
                sentence_pre_word_id_list_bag_all_dev_batch = np.array(sentence_pre_word_id_list_bag_all_dev_batch)
                word_position_list_bag_all_dev_batch = np.array(word_position_list_bag_all_dev_batch)
                label_bag_all_dev_batch = np.array(label_bag_all_dev_batch)
                adjacent_words_id_bag_all_dev_batch = np.array(adjacent_words_id_bag_all_dev_batch)
                trigger_parts_index_bag_all_dev_batch = np.array(trigger_parts_index_bag_all_dev_batch)
                bag_flags_all_dev_batch = np.array(bag_flags_all_dev_batch)

                feed_dict = {
                    DMcnn_trigger_bag.input_sentence: sentence_pre_word_id_list_bag_all_dev_batch,
                    DMcnn_trigger_bag.input_word_position: word_position_list_bag_all_dev_batch,
                    DMcnn_trigger_bag.input_role: label_bag_all_dev_batch,
                    DMcnn_trigger_bag.input_parts_indexs: trigger_parts_index_bag_all_dev_batch,
                    DMcnn_trigger_bag.input_adjacent_words: adjacent_words_id_bag_all_dev_batch,
                    DMcnn_trigger_bag.bag_num: bag_num_dev_batch,
                    DMcnn_trigger_bag.input_flags: bag_flags_all_dev_batch,
                    DMcnn_trigger_bag.input_neg_label: sample_label_neg_batch_dev,
                    DMcnn_trigger_bag.scores_lookup_add: scores_lookup_add_dev,
                    DMcnn_trigger_bag.dropout_keep_prob: FLAGS.dropout_keep_prob}

                loss, accuracy, pos_pre, pos_pre_all, pos_pre_num, pos_all_num, precision, recall, F1, scores_max, scores_max_id, prediction_labels, input_labels, scores_flat = sess.run(
                    [DMcnn_trigger_bag.loss, DMcnn_trigger_bag.accuracy,
                     DMcnn_trigger_bag.positive_correct_prediction,
                     DMcnn_trigger_bag.positive_prediction,
                     DMcnn_trigger_bag.positive_correct_prediction_num,
                     DMcnn_trigger_bag.positive_correct_all_num,
                     DMcnn_trigger_bag.precision, DMcnn_trigger_bag.recall_value,
                     DMcnn_trigger_bag.F1_value,
                     DMcnn_trigger_bag.scores_max_score,
                     DMcnn_trigger_bag.scores_lookup_ids,
                     DMcnn_trigger_bag.prediction,
                     DMcnn_trigger_bag.org_label,
                     DMcnn_trigger_bag.scores_flat], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                loss_dev = loss_dev + loss
                acc_dev = acc_dev + accuracy
                pos_pre_all_dev = pos_pre_all_dev + pos_pre_all
                pos_pre_num_dev = pos_pre_num_dev + pos_pre_num
                pos_all_num_dev = pos_all_num_dev + pos_all_num

                # =====================================Pre info=======================================================
                # word_concated = concat_word(word_bag_all_dev_batch)
                # word_concated_ids = convert_word2id(vocabulary_id_dict, word_concated)
                # print(scores_max)
                # print(scores_max_id)
                # print(word_concated_ids)
                scores_max_ids = []
                scores_maxs = []
                org_labels = []
                pre_labels = []
                dev_sentences_word_num_batch = list(dev_sentences_word_num_batch)
                multi_max_words_ids_dev_batch = []
                bag_word_add = 0
                for i in range(len(dev_sentences_word_num_batch)):
                    bag_max_words_pre_sentence = []
                    bag_max_scores_id_pre_sentence = get_max_score_pre_sentence(scores_flat[i], dev_sentences_word_num_batch[i], bag_word_add)
                    # bag_max_word_pre_sentence = tf.nn.embedding_lookup(word_concated, bag_max_scores_id_pre_sentence)
                    # bag_max_words_pre_sentence = sess.run(bag_max_word_pre_sentence)
                    for j in range(len(bag_max_scores_id_pre_sentence)):
                        bag_max_words_pre_sentence.append(word_bag_all_dev_batch[int(bag_max_scores_id_pre_sentence[j])])
                    multi_max_words_ids_dev_batch.append(bag_max_words_pre_sentence)
                    scores_max_ids.append(int(scores_max_id[i]))
                    scores_maxs.append(float(scores_max[i]))
                    org_labels.append(input_labels[i])
                    pre_labels.append(prediction_labels[i])
                    bag_word_add += 100

                max_words = []
                for k in range(len(scores_max_ids)):
                    max_words.append(word_bag_all_dev_batch[int(scores_max_ids[k])])
                write_bag_predict_info(dev_bag_org_ids_batch, dev_pos_neg_bag_org_ids_batch, scores_maxs, pre_labels,
                                       org_labels, scores_max_ids, max_words, multi_max_words_ids_dev_batch, Params.bag_dev_predict_info_path_1)
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


def cut_bag_word(word_bag_all, sentence_pre_word_id_list_bag_all, word_position_list_bag_all, adjacent_words_id_bag_all, trigger_parts_index_bag_all, bag_word_length):
    """
    Cut bag word
    :param sentence_pre_word_id_list_bag_all:
    :param word_position_list_bag_all:
    :param adjacent_words_id_bag_all:
    :param trigger_parts_index_bag_all:
    :return:
    """
    word_bag_all_cut = []
    sentence_pre_word_id_list_bag_all_cut = []
    word_position_list_bag_all_cut = []
    adjacent_words_id_bag_all_cut = []
    trigger_parts_index_bag_all_cut = []
    if len(sentence_pre_word_id_list_bag_all) <= bag_word_length:
        num = bag_word_length - len(sentence_pre_word_id_list_bag_all)

        word_bag_all_cut = word_bag_all
        sentence_pre_word_id_list_bag_all_cut = sentence_pre_word_id_list_bag_all
        word_position_list_bag_all_cut = word_position_list_bag_all
        adjacent_words_id_bag_all_cut = adjacent_words_id_bag_all
        trigger_parts_index_bag_all_cut = trigger_parts_index_bag_all

        for i in range(num):
            word_bag_all_cut.append(word_bag_all[0])
            sentence_pre_word_id_list_bag_all_cut.append(sentence_pre_word_id_list_bag_all[0])
            word_position_list_bag_all_cut.append(word_position_list_bag_all[0])
            adjacent_words_id_bag_all_cut.append(adjacent_words_id_bag_all[0])
            trigger_parts_index_bag_all_cut.append(trigger_parts_index_bag_all[0])
    else:
        for i in range(bag_word_length):
            word_bag_all_cut.append(word_bag_all[i])
            sentence_pre_word_id_list_bag_all_cut.append(sentence_pre_word_id_list_bag_all[i])
            word_position_list_bag_all_cut.append(word_position_list_bag_all[i])
            adjacent_words_id_bag_all_cut.append(adjacent_words_id_bag_all[i])
            trigger_parts_index_bag_all_cut.append(trigger_parts_index_bag_all[i])

    return word_bag_all_cut, sentence_pre_word_id_list_bag_all_cut, word_position_list_bag_all_cut, adjacent_words_id_bag_all_cut, trigger_parts_index_bag_all_cut


def split_sample(combine_sample):
    """
    Split Sample
    :param combine_sample:
    :return: split sample
    """
    word_list_bag, sentence_pre_word_id_list_bag, word_position_list_bag, label_bag, adjacent_words_id_bag, \
    trigger_parts_index_bag, bag_flags, bag_positive_org_ids, pos_neg_bag_id_list, sentences_word_num = zip(*combine_sample)
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

    return word_list_bag, sentence_pre_word_id_list_bag, word_position_list_bag, label_bag, adjacent_words_id_bag, trigger_parts_index_bag, bag_flags, bag_positive_org_ids, pos_neg_bag_id_list, sentences_word_num


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


def concat_word(word_list_bag):
    """
    Concat word
    :param word_list_bag:
    :return: Concated word
    """
    concat_word = []
    # print(word_list_bag)
    for word in word_list_bag:
        # print(word)
        word_temp = []
        word_temp.append(word)
        concat_word.append(word_temp)

    return concat_word


def convert_word2id(vocabulary_id_dict, word_concated):
    """
    Convert word 2 ids
    :param vocabulary_id_dict:
    :param word_concated:
    :return: word 2 ids
    """
    word_concated_ids = []
    # print(word_concated)
    for words in word_concated:
        # print(word_concated)
        for word in words:
            word_concated_id = []
            word_concated_id.append(int(vocabulary_id_dict[word]))
        word_concated_ids.append(word_concated_id)
    # print(word_concated_ids)
    return word_concated_ids


def get_max_score_pre_sentence(scores_pre_bag, bag_word_nums_pre_sentence, bag_word_add):
    """
    Get max score pre sentence in a bag.
    :param scores_pre_bag:
    :param bag_word_nums_pre_sentence:
    :param bag_word_add:
    :return: Max score word id in sentence.
    """
    scores_pre_bag = list(scores_pre_bag)
    # print(len(scores_pre_bag))
    scores_pre_sentence = []
    num_before = 0
    num_next = 0
    for num in bag_word_nums_pre_sentence:
        # print(num)
        num_next += num
        score_list = scores_pre_bag[int(num_before * 22):int(num_next * 22)]
        scores_pre_sentence.append(score_list)
        num_before += num
        # print(len(score_list))
        # print(scores_pre_sentence)

    # print(len(scores_pre_sentence))
    bag_max_scores_id_pre_sentence = []
    for i in range(len(scores_pre_sentence)):
        if i == 0:
            bag_max_scores_id_pre_sentence.append(int(np.argmax(scores_pre_sentence[i]) / 22 + bag_word_add))
        elif i > 0:
            bag_max_scores_id_pre_sentence.append(int(np.argmax(scores_pre_sentence[i]) / 22 + bag_word_add + bag_word_nums_pre_sentence[i - 1]))
    # print(int(np.argmax(scores)))
    # print(bag_max_scores_id_pre_sentence)

    return bag_max_scores_id_pre_sentence


def write_bag_predict_info(bag_org_ids, pos_neg_bag_org_ids, scores_max, pre_labels, org_labels, scores_max_ids, max_words, scores_max_mul_ids, output_filename):
    """
    Process bag predict information for sort P-R curve.
    :param bag_org_ids:
    :param pos_neg_bag_org_ids:
    :param scores_max:
    :param pre_labels:
    :param org_labels:
    :param scores_max_ids: Max scores word id in sentence sequence
    :param max_words: Max scores word in vocabulary
    :param scores_max_mul_ids:Multi max scores word id about every sentence in vocabulary
    :return: Process and write into file.
    """
    print("Write result info into file...")
    with open(output_filename, "a", encoding="utf-8") as output_file:
        for i in range(len(bag_org_ids)):
            output_file.write(str(bag_org_ids[i]) + " ")
            output_file.write(str(pos_neg_bag_org_ids[i]) + " ")
            output_file.write(str(scores_max[i]) + " ")
            output_file.write(str(pre_labels[i]) + " ")
            output_file.write(str(org_labels[i]) + " ")
            output_file.write(str(scores_max_ids[i]) + " ")
            output_file.write(str(max_words[i]) + " ")
            for j in range(len(scores_max_mul_ids[i])):
                output_file.write(str(scores_max_mul_ids[i][j]) + " ")
            output_file.write("\n")
        output_file.close()


if __name__ == '__main__':
    tagger_trigger_bag()