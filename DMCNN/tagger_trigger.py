# -*- coding: utf-8 -*-

"""
Identify the trigger in a input sentence
Author: ZUO XINYU
Data: 08/04/2017
"""

import tensorflow as tf
from DMCNN import DMCNN
import Params
import tensor_trigger_val
import tensor_trigger
import numpy as np
import dataProcess_trigger_val

# Parameters
# =========================================================================

# Data load parameters
tf.flags.DEFINE_string("positive_sample_path", Params.test_positive_sample_path, "Positive sample file")
tf.flags.DEFINE_string("negative_sample_path", Params.test_negative_sample_path, "Negative sample file")

# Model parameters
tf.flags.DEFINE_integer("sentence_length", Params.sentence_length, "The length of sentence")
tf.flags.DEFINE_integer("num_role", Params.num_role, "The number of role")
tf.flags.DEFINE_integer("vocabulary_size", Params.vocabulary_size, "The size of vocabulary")
tf.flags.DEFINE_integer("embedding_size", Params.embedding_size, "The size of embedding")
tf.flags.DEFINE_integer("position_size", Params.position_size, "The size of word position")
tf.flags.DEFINE_integer("position_embedding_size", Params.position_embedding_size,
                        "The size of word position embedding")
tf.flags.DEFINE_integer("event_type_embedding_size", Params.event_type_embedding_size,
                        "The size of event type embedding")
tf.flags.DEFINE_integer("filter_size", Params.filter_size, "The size of filter")
tf.flags.DEFINE_integer("num_filters", Params.num_filters, "the number of filter")
tf.flags.DEFINE_integer("num_epochs", Params.num_epochs, "The number of epoch")

# Training parameters
tf.flags.DEFINE_integer("batch_size", Params.batch_size, "The size of batch")
tf.flags.DEFINE_float("dropout_keep_prob", Params.dropout_keep_prob, "The dropout of training")
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

# Data load
# ===================================single sentence=================================

# vocabulary, vocabulary_id_dict, _, _, _ = dataProcess_trigger_val.create_vocabulary_dict(Params.wiki_sentence_annotated_with_trigger_path,
#                                                                      Params.sentence_length)
# word_all_sentence_list = dataProcess_trigger_val.get_word_line("./m.0100cyqx,m.026xgfj.tsv", Params.sentence_length)
# print(word_all_sentence_list)
# for word_line in word_all_sentence_list:
#     word_position_list, sentence_pre_word, word_list_pre_sentence = dataProcess_trigger_val.get_word_position_and_sentence(word_line)
#     word_id_pre_sentence = dataProcess_trigger_val.get_word_id(vocabulary_id_dict, word_list_pre_sentence)
#     sentence_id_pre_word = dataProcess_trigger_val.get_sentence_id_pre_word(vocabulary_id_dict, sentence_pre_word)
#     word_adjacent_words_list = dataProcess_trigger_val.get_adjacent_words(word_line)
#     words_adjacent_word_id_sentence = dataProcess_trigger_val.get_adjacent_words_id(vocabulary_id_dict, word_adjacent_words_list)
#
# sample_word, sample_sentence, sample_position, label, sample_parts_indexs, smaple_adjacent_words = tensor_trigger_val.get_tensor(word_id_pre_sentence, sentence_id_pre_word, word_position_list,
#                                   words_adjacent_word_id_sentence, Params.sentence_length)
#
# sample_word = np.array(sample_word)
# sample_sentence = np.array(sample_sentence)
# sample_position = np.array(sample_position)
# label = np.array(label)
# sample_parts_indexs = np.array(sample_parts_indexs)
# smaple_adjacent_words = np.array(smaple_adjacent_words)

# print(tf.shape(sample_word))
# print(tf.shape(sample_sentence))
# print(tf.shape(sample_position))
# print(tf.shape(label))
# print(tf.shape(sample_parts_indexs))
# print(tf.shape(smaple_adjacent_words))

# =================================positive samples test===============================================
positive_sample_word, positive_sample_sentence_test, positive_sample_position_test, positive_sample_label_test, \
positive_sample_parts_indexs_test, positive_sample_adjacent_words_test, positive_sample_flags_test = tensor_trigger.get_tensor(FLAGS.positive_sample_path, FLAGS.negative_sample_path, Params.sentence_length, 1000)

positive_sample_adjacent_words_test = np.array(positive_sample_adjacent_words_test)
positive_sample_sentence_test = np.array(positive_sample_sentence_test)
positive_sample_position_test = np.array(positive_sample_position_test)
positive_sample_label_test = np.array(positive_sample_label_test)
positive_sample_adjacent_words_test = np.array(positive_sample_adjacent_words_test)
positive_sample_parts_indexs_test = np.array(positive_sample_parts_indexs_test)
positive_sample_flags_test = np.array(positive_sample_flags_test)

sample_label_neg_batch_test = []
for i in range(len(positive_sample_flags_test)):
    sample_neg_flags_temp = [int(0)] * 21
    sample_neg_flags_temp.append(int(1))
    sample_label_neg_batch_test.append(sample_neg_flags_temp)
sample_label_neg_batch_test = np.array(sample_label_neg_batch_test)

# Tagger trigger
# ====================================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        DMcnn = DMCNN(
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
            gpu_flags="/gpu:0"
        )

        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, "./trigger/1505732546/checkpoints/model-945000")

        #sess.run(tf.global_variables_initializer())

        # sample_word, sample_sentence, sample_position, label, sample_parts_indexs, smaple_adjacent_words = sess.run(
        #     [sample_word, sample_sentence, sample_position, label, sample_parts_indexs, smaple_adjacent_words])

        # feed_dict = {DMcnn.input_sentence: sample_sentence,
        #              DMcnn.input_word_position: sample_position, DMcnn.input_role: label,
        #              DMcnn.input_parts_indexs: sample_parts_indexs,
        #              DMcnn.input_adjacent_words: smaple_adjacent_words,
        #              DMcnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
        #
        # loss, accuracy, prediction, scores = sess.run([DMcnn.loss, DMcnn.accuracy, DMcnn.prediction, DMcnn.scores], feed_dict)
        # print(loss)
        # print(accuracy)
        # print(prediction)
        # print(scores)

        feed_dict = {DMcnn.input_sentence: positive_sample_sentence_test,
                     DMcnn.input_word_position: positive_sample_position_test,
                     DMcnn.input_role: positive_sample_label_test,
                     DMcnn.input_parts_indexs: positive_sample_parts_indexs_test,
                     DMcnn.input_adjacent_words: positive_sample_adjacent_words_test,
                     DMcnn.input_flags: positive_sample_flags_test,
                     DMcnn.input_neg_label: sample_label_neg_batch_test,
                     DMcnn.dropout_keep_prob: FLAGS.dropout_keep_prob}

        loss, accuracy, recall, F1, prediction, scores = sess.run(
            [DMcnn.loss, DMcnn.accuracy, DMcnn.recall_value, DMcnn.F1_value, DMcnn.prediction, DMcnn.scores], feed_dict)
        print(loss)
        print(accuracy)
        print(recall)
        print(F1)
        print(prediction)
        print(scores)
