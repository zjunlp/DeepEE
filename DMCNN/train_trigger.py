# -*- coding: utf-8 -*-

from DMCNN import DMCNN
import Params
import tensor_trigger
import random
import time
import datetime
import os
import tensorflow as tf
import numpy as np

"""
Train file for DMCNN
Author: ZUO XINYU
Date: 07/25/2017
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# tf_config = tf.ConfigProto(allow_soft_placement=True,
#                            log_device_placement=True)
# tf_config.gpu_options.allow_growth = True

# Parameters
# =========================================================================

# Data load parameters
tf.flags.DEFINE_string("train_positive_sample_path", Params.positive_sample_path, "Train Positive sample file")
tf.flags.DEFINE_string("train_negative_sample_path", Params.negative_sample_path, "Train Negative sample file")

tf.flags.DEFINE_string("test_positive_sample_path", Params.test_positive_sample_path, "Test Positive sample file")
tf.flags.DEFINE_string("test_negative_sample_path", Params.test_negative_sample_path, "Test Negative sample file")

tf.flags.DEFINE_string("dev_positive_sample_path", Params.dev_positive_sample_path, "Dev Positive sample file")
tf.flags.DEFINE_string("dev_negative_sample_path", Params.dev_negative_sample_path, "Dev Negative sample file")
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
tf.flags.DEFINE_boolean("log_device_placement", True, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data preparation
# ====================================================================

# Load data
print(str(Params.cross_fold))
print("Loading train data...")
sample_word_train, sample_sentence_train, sample_position_train, label_train, sample_parts_indexs_train, \
sample_adjacent_words_train, sample_flags_train = tensor_trigger.get_tensor(FLAGS.train_positive_sample_path, FLAGS.train_negative_sample_path, FLAGS.sentence_length, 63000)
sample_total = len(label_train)
print("Train data num %d" % len(label_train))

print("Loading test data...")
sample_word_test, sample_sentence_test, sample_position_test, label_test, sample_parts_indexs_test, \
sample_adjacent_words_test, sample_flags_test = tensor_trigger.get_tensor(FLAGS.test_positive_sample_path, FLAGS.test_negative_sample_path, FLAGS.sentence_length, 8000)
print("Train data num %d" % len(label_test))

print("Loading dev data...")
sample_word_dev, sample_sentence_dev, sample_position_dev, label_dev, sample_parts_indexs_dev, \
sample_adjacent_words_dev, sample_flags_dev = tensor_trigger.get_tensor(FLAGS.dev_positive_sample_path, FLAGS.dev_negative_sample_path, FLAGS.sentence_length, 7000)
print("Train data num %d" % len(label_dev))

print("Shuffle sample...")
combine_sample_train = list(zip(sample_word_train, sample_sentence_train, sample_position_train, label_train, sample_parts_indexs_train, sample_adjacent_words_train, sample_flags_train))
random.shuffle(combine_sample_train)

combine_sample_test = list(zip(sample_word_test, sample_sentence_test, sample_position_test, label_test, sample_parts_indexs_test, sample_adjacent_words_test, sample_flags_test))
random.shuffle(combine_sample_test)

combine_sample_dev = list(zip(sample_word_dev, sample_sentence_dev, sample_position_dev, label_dev, sample_parts_indexs_dev, sample_adjacent_words_dev, sample_flags_dev))
random.shuffle(combine_sample_dev)


def train(Round):
    # Training
    # ====================================================================
    with tf.Graph().as_default(), tf.device('/gpu:3'):
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement
        )
        session_conf.gpu_options.allow_growth = True
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
                gpu_flags='/gpu:2'
            )

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(DMcnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity(optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grads_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grads_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory foe model and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "trigger", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            # Train summaries
            loss_summary = tf.summary.scalar("loss", DMcnn.loss)
            acc_summary = tf.summary.scalar("accuracy", DMcnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.checkpoint_pre)

            # results data directory
            results_dir = os.path.abspath(os.path.join(out_dir, "results"))
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            for i in range(Round):
                # results dir
                results_prefix = os.path.join(results_dir, "round-" + str(i + 1) + ".log")
                # Load data
                sample_word_train, sample_sentence_train, sample_position_train, label_train, sample_parts_indexs_train, \
                sample_adjacent_words_train, sample_flags_train = convert_sample(combine_sample_train)

                sample_word_test, sample_sentence_test, sample_position_test, label_test, sample_parts_indexs_test, \
                sample_adjacent_words_test, sample_flags_test = convert_sample(combine_sample_test)

                sample_word_dev, sample_sentence_dev, sample_position_dev, label_dev, sample_parts_indexs_dev, \
                sample_adjacent_words_dev, sample_flags_dev = convert_sample(combine_sample_dev)

                # Train
                batches = batch_iter(
                    list(zip(sample_word_train, sample_sentence_train, sample_position_train, label_train,
                             sample_parts_indexs_train, sample_adjacent_words_train, sample_flags_train)), FLAGS.batch_size, 1)

                batch_num = 0

                # Write results into file
                with open(results_prefix, 'w', encoding='utf-8') as results_file:
                    for batch in batches:
                        batch_num += 1
                        print("batch number %d" % batch_num)
                        sample_word_batch, sample_sentence_batch, sample_position_batch, label_batch, \
                        sample_parts_indexs_batch, smaple_adjacent_words_batch, sample_flags_batch = zip(*batch)

                        sample_label_neg_batch = []
                        for i in range(len(label_batch)):
                            sample_neg_flags_temp = [int(0)] * 21
                            sample_neg_flags_temp.append(int(1))
                            sample_label_neg_batch.append(sample_neg_flags_temp)
                        sample_label_neg_batch = np.array(sample_label_neg_batch)

                        feed_dict = {DMcnn.input_sentence: sample_sentence_batch,
                                     DMcnn.input_word_position: sample_position_batch, DMcnn.input_role: label_batch,
                                     DMcnn.input_parts_indexs: sample_parts_indexs_batch,
                                     DMcnn.input_adjacent_words: smaple_adjacent_words_batch,
                                     DMcnn.input_flags: sample_flags_batch,
                                     DMcnn.input_neg_label: sample_label_neg_batch,
                                     DMcnn.dropout_keep_prob: FLAGS.dropout_keep_prob}

                        _, step, summaries, loss, accuracy, pos_pre, pos_pre_all, pos_pre_num, pos_all_num, precision, recall, F1 = sess.run(
                            [train_op, global_step, train_summary_op, DMcnn.loss, DMcnn.accuracy,
                                DMcnn.positive_correct_prediction, DMcnn.positive_prediction,
                                DMcnn.positive_correct_prediction_num, DMcnn.positive_correct_all_num, DMcnn.precision,
                                DMcnn.recall_value, DMcnn.F1_value], feed_dict)
                        time_str = datetime.datetime.now().isoformat()
                        print("train: {}: step {}, loss {:g}, acc {:g}, pre {:g}, recall {:g}, F1 {:g}".format(time_str, step, loss, accuracy, precision, recall, F1))

                        results_file.write("train: {}: step {}, loss {:g}, acc {:g}, pre {:g}, recall {:g}, F1 {:g}".format(time_str, step, loss, accuracy, precision, recall, F1))
                        results_file.write('\n')
                        # print('\n')
                        # print(pos_pre)
                        # print(pos_pre_all)
                        # print(pos_pre_num)
                        # print(pos_all_num)
                        # print('\n')
                        if train_summary_writer:
                            train_summary_writer.add_summary(summaries, step)

                        # current_step = tf.train.global_step(sess, global_step)
                        if step % FLAGS.evaluate_pre == 0:
                            print("\nEvaluation:")
                            # Dev
                            # ==========================================
                            batch_dev_num = 0
                            loss_dev = 0.0
                            acc_dev = 0.0
                            pos_pre_all_dev = 0
                            pos_pre_num_dev = 0
                            pos_all_num_dev = 0
                            # ============================================
                            batches_dev = batch_iter(
                                list(zip(sample_word_dev, sample_sentence_dev, sample_position_dev, label_dev,
                                         sample_parts_indexs_dev, sample_adjacent_words_dev, sample_flags_dev)),
                                5 * FLAGS.batch_size, 0)
                            # ==============================================
                            for batch_dev in batches_dev:
                                batch_dev_num += 1
                                sample_word_batch_dev, sample_sentence_batch_dev, sample_position_batch_dev, label_batch_dev, \
                                sample_parts_indexs_batch_dev, smaple_adjacent_words_batch_dev, sample_flags_batch_dev = zip(*batch_dev)

                                sample_label_neg_batch_dev = []
                                for i in range(len(label_batch_dev)):
                                    sample_neg_flags_temp = [int(0)] * 21
                                    sample_neg_flags_temp.append(int(1))
                                    sample_label_neg_batch_dev.append(sample_neg_flags_temp)
                                sample_label_neg_batch_dev = np.array(sample_label_neg_batch_dev)

                                feed_dict = {DMcnn.input_sentence: sample_sentence_batch_dev,
                                             DMcnn.input_word_position: sample_position_batch_dev, DMcnn.input_role: label_batch_dev,
                                             DMcnn.input_parts_indexs: sample_parts_indexs_batch_dev,
                                             DMcnn.input_adjacent_words: smaple_adjacent_words_batch_dev,
                                             DMcnn.input_flags: sample_flags_batch_dev,
                                             DMcnn.input_neg_label: sample_label_neg_batch_dev,
                                             DMcnn.dropout_keep_prob: FLAGS.dropout_keep_prob}

                                step, summaries, loss, accuracy, pos_pre, pos_pre_all, pos_pre_num, pos_all_num, precision, recall, F1 = sess.run(
                                    [global_step, dev_summary_op, DMcnn.loss, DMcnn.accuracy,
                                        DMcnn.positive_correct_prediction, DMcnn.positive_prediction,
                                        DMcnn.positive_correct_prediction_num, DMcnn.positive_correct_all_num, DMcnn.precision,
                                        DMcnn.recall_value, DMcnn.F1_value], feed_dict)
                                time_str = datetime.datetime.now().isoformat()
                                loss_dev = loss_dev + loss
                                acc_dev = acc_dev + accuracy
                                pos_pre_all_dev = pos_pre_all_dev + pos_pre_all
                                pos_pre_num_dev = pos_pre_num_dev + pos_pre_num
                                pos_all_num_dev = pos_all_num_dev + pos_all_num

                            if dev_summary_writer:
                                dev_summary_writer.add_summary(summaries, step)
                            print("")

                            print(batch_dev_num)
                            loss = loss_dev/batch_dev_num
                            accuracy = acc_dev/batch_dev_num
                            precision = pos_pre_num_dev / pos_pre_all_dev
                            recall = pos_pre_num_dev / pos_all_num_dev
                            F1 = (2 * precision * recall) / (precision + recall)

                            print("Dev: {}: step {}, loss {:g}, acc {:g}, pre {:g}, recall {:g}, F1 {:g}".format(time_str, step, loss, accuracy, precision, recall, F1))
                            results_file.write("Dev: {}: step {}, loss {:g}, acc {:g}, pre {:g}, recall {:g}, F1 {:g}".format(time_str, step, loss, accuracy, precision, recall, F1))
                            results_file.write('\n')
                            results_file.write('\n')
                            print('\n')
                            print(pos_pre_all_dev)
                            print(pos_pre_num_dev)
                            print(pos_all_num_dev)
                            print('\n')
                        if step % FLAGS.checkpoint_pre == 0:
                            path = saver.save(sess, checkpoint_prefix, global_step=step)
                            print("Saved model checkpoint to {}\n".format(path))
                    # ===================Dev=========================
                    batch_dev_num = 0
                    loss_dev = 0.0
                    acc_dev = 0.0
                    pos_pre_all_dev = 0
                    pos_pre_num_dev = 0
                    pos_all_num_dev = 0
                    batches_dev = batch_iter(list(zip(sample_word_dev, sample_sentence_dev, sample_position_dev, label_dev,
                                 sample_parts_indexs_dev, sample_adjacent_words_dev, sample_flags_dev)),5 * FLAGS.batch_size, 0)
                    # ==============================================
                    for batch_dev in batches_dev:
                        batch_dev_num += 1
                        sample_word_batch_dev, sample_sentence_batch_dev, sample_position_batch_dev, label_batch_dev, \
                        sample_parts_indexs_batch_dev, smaple_adjacent_words_batch_dev, sample_flags_batch_dev = zip(*batch_dev)

                        sample_label_neg_batch_dev = []
                        for i in range(len(label_batch_dev)):
                            sample_neg_flags_temp = [int(0)] * 21
                            sample_neg_flags_temp.append(int(1))
                            sample_label_neg_batch_dev.append(sample_neg_flags_temp)
                        sample_label_neg_batch_dev = np.array(sample_label_neg_batch_dev)

                        feed_dict = {DMcnn.input_sentence: sample_sentence_batch_dev,
                                        DMcnn.input_word_position: sample_position_batch_dev,
                                        DMcnn.input_role: label_batch_dev,
                                        DMcnn.input_parts_indexs: sample_parts_indexs_batch_dev,
                                        DMcnn.input_adjacent_words: smaple_adjacent_words_batch_dev,
                                        DMcnn.input_flags: sample_flags_batch_dev,
                                        DMcnn.input_neg_label: sample_label_neg_batch_dev,
                                        DMcnn.dropout_keep_prob: FLAGS.dropout_keep_prob}

                        step, summaries, loss, accuracy, pos_pre, pos_pre_all, pos_pre_num, pos_all_num, precision, recall, F1 = sess.run(
                            [global_step, dev_summary_op, DMcnn.loss, DMcnn.accuracy,
                                DMcnn.positive_correct_prediction, DMcnn.positive_prediction,
                                DMcnn.positive_correct_prediction_num, DMcnn.positive_correct_all_num,
                                DMcnn.precision,
                                DMcnn.recall_value, DMcnn.F1_value], feed_dict)
                        time_str = datetime.datetime.now().isoformat()
                        loss_dev = loss_dev + loss
                        acc_dev = acc_dev + accuracy
                        pos_pre_all_dev = pos_pre_all_dev + pos_pre_all
                        pos_pre_num_dev = pos_pre_num_dev + pos_pre_num
                        pos_all_num_dev = pos_all_num_dev + pos_all_num

                    if dev_summary_writer:
                        dev_summary_writer.add_summary(summaries, step)
                    print("")

                    print(batch_dev_num)
                    loss = loss_dev / batch_dev_num
                    accuracy = acc_dev / batch_dev_num
                    precision = pos_pre_num_dev / pos_pre_all_dev
                    recall = pos_pre_num_dev / pos_all_num_dev
                    F1 = (2 * precision * recall) / (precision + recall)

                    print("Dev: {}: step {}, loss {:g}, acc {:g}, pre {:g}, recall {:g}, F1 {:g}".format(time_str, step, loss, accuracy, precision, recall, F1))
                    results_file.write("Dev: {}: step {}, loss {:g}, acc {:g}, pre {:g}, recall {:g}, F1 {:g}".format(time_str, step, loss, accuracy, precision, recall, F1))
                    results_file.write('\n')
                    results_file.write('\n')
                    print('\n')
                    print(pos_pre_all_dev)
                    print(pos_pre_num_dev)
                    print(pos_all_num_dev)
                    print('\n')
                    # ===================Test=========================
                    batch_test_num = 0
                    loss_test = 0.0
                    acc_test = 0.0
                    pos_pre_all_test = 0
                    pos_pre_num_test = 0
                    pos_all_num_test = 0
                    batches_test = batch_iter(list(zip(sample_word_test, sample_sentence_test, sample_position_test, label_test,
                                 sample_parts_indexs_test, sample_adjacent_words_test, sample_flags_test)), 5 * FLAGS.batch_size, 0)
                    # ==============================================
                    for batche_test in batches_test:
                        batch_test_num += 1
                        sample_word_batch_test, sample_sentence_batch_test, sample_position_batch_test, label_batch_test, \
                        sample_parts_indexs_batch_test, smaple_adjacent_words_batch_test, sample_flags_batch_test = zip(*batche_test)

                        sample_label_neg_batch_test = []
                        for i in range(len(label_batch_test)):
                            sample_neg_flags_temp = [int(0)] * 21
                            sample_neg_flags_temp.append(int(1))
                            sample_label_neg_batch_test.append(sample_neg_flags_temp)
                        sample_label_neg_batch_test = np.array(sample_label_neg_batch_test)

                        feed_dict = {DMcnn.input_sentence: sample_sentence_batch_test,
                                     DMcnn.input_word_position: sample_position_batch_test,
                                     DMcnn.input_role: label_batch_test,
                                     DMcnn.input_parts_indexs: sample_parts_indexs_batch_test,
                                     DMcnn.input_adjacent_words: smaple_adjacent_words_batch_test,
                                     DMcnn.input_flags: sample_flags_batch_test,
                                     DMcnn.input_neg_label: sample_label_neg_batch_test,
                                     DMcnn.dropout_keep_prob: FLAGS.dropout_keep_prob}

                        step, summaries, loss, accuracy, pos_pre, pos_pre_all, pos_pre_num, pos_all_num, precision, recall, F1 = sess.run(
                            [global_step, dev_summary_op, DMcnn.loss, DMcnn.accuracy,
                             DMcnn.positive_correct_prediction, DMcnn.positive_prediction,
                             DMcnn.positive_correct_prediction_num, DMcnn.positive_correct_all_num,
                             DMcnn.precision,
                             DMcnn.recall_value, DMcnn.F1_value], feed_dict)
                        time_str = datetime.datetime.now().isoformat()
                        loss_test = loss_test + loss
                        acc_test = acc_test + accuracy
                        pos_pre_all_test = pos_pre_all_test + pos_pre_all
                        pos_pre_num_test = pos_pre_num_test + pos_pre_num
                        pos_all_num_test = pos_all_num_test + pos_all_num

                    if dev_summary_writer:
                        dev_summary_writer.add_summary(summaries, step)
                    print("")

                    print(batch_dev_num)
                    loss = loss_dev / batch_test_num
                    accuracy = acc_dev / batch_test_num
                    precision = pos_pre_num_test / pos_pre_all_test
                    recall = pos_pre_num_test / pos_all_num_test
                    F1 = (2 * precision * recall) / (precision + recall)

                    print("Test: {}: step {}, loss {:g}, acc {:g}, pre {:g}, recall {:g}, F1 {:g}".format(time_str, step, loss, accuracy, precision, recall, F1))
                    results_file.write("Test: {}: step {}, loss {:g}, acc {:g}, pre {:g}, recall {:g}, F1 {:g}".format(time_str, step, loss, accuracy, precision, recall, F1))
                    results_file.write('\n')
                    results_file.write('\n')
                    print('\n')
                    print(pos_pre_all_test)
                    print(pos_pre_num_test)
                    print(pos_all_num_test)
                    print('\n')
                results_file.close()


def split_sample(combine_sample, Round):
    """
    Split Sample
    :param combine_sample:
    :return: split sample
    """
    sample_word_tensor, sample_sentence_tensor, sample_position_tensor, label_tensor, sample_parts_indexs_tensor, \
    sample_adjacent_words_tensor, sample_flags_tensor = zip(*combine_sample)
    print(len(sample_word_tensor))

    # Train and test data spilt
    # ==================================================================
    sample_word_dev = sample_word_tensor[
                      int((Round - 1) * sample_total / FLAGS.cross_fold): int(Round * sample_total / FLAGS.cross_fold)]
    sample_sentence_dev = sample_sentence_tensor[
                          int((Round - 1) * sample_total / FLAGS.cross_fold): int(
                              Round * sample_total / FLAGS.cross_fold)]
    sample_position_dev = sample_position_tensor[
                          int((Round - 1) * sample_total / FLAGS.cross_fold): int(
                              Round * sample_total / FLAGS.cross_fold)]
    label_dev = label_tensor[
                int((Round - 1) * sample_total / FLAGS.cross_fold): int(Round * sample_total / FLAGS.cross_fold)]
    sample_parts_indexs_dev = sample_parts_indexs_tensor[
                              int((Round - 1) * sample_total / FLAGS.cross_fold): int(
                                  Round * sample_total / FLAGS.cross_fold)]
    smaple_adjacent_words_dev = sample_adjacent_words_tensor[
                                int((Round - 1) * sample_total / FLAGS.cross_fold): int(
                                    Round * sample_total / FLAGS.cross_fold)]
    sample_flags_dev = sample_flags_tensor[
                       int((Round - 1) * sample_total / FLAGS.cross_fold): int(Round * sample_total / FLAGS.cross_fold)]

    sample_word_dev = np.array(sample_word_dev)
    sample_sentence_dev = np.array(sample_sentence_dev)
    sample_position_dev = np.array(sample_position_dev)
    label_dev = np.array(label_dev)
    smaple_adjacent_words_dev = np.array(smaple_adjacent_words_dev)
    sample_parts_indexs_dev = np.array(sample_parts_indexs_dev)
    sample_flags_dev = np.array(sample_flags_dev)
    # print(sample_parts_indexs_dev)
    # sample_parts_indexs_dev = tf.transpose(sample_parts_indexs_dev, [1, 2, 0])
    # print(sample_parts_indexs_dev)


    sample_word_train = sample_word_tensor[int(Round * sample_total / FLAGS.cross_fold):] + \
                        sample_word_tensor[:int((Round - 1) * sample_total / FLAGS.cross_fold)]
    sample_sentence_train = sample_sentence_tensor[int(Round * sample_total / FLAGS.cross_fold):] + \
                            sample_sentence_tensor[:int((Round - 1) * sample_total / FLAGS.cross_fold)]
    sample_position_train = sample_position_tensor[int(Round * sample_total / FLAGS.cross_fold):] + \
                            sample_position_tensor[:int((Round - 1) * sample_total / FLAGS.cross_fold)]
    label_train = label_tensor[int(Round * sample_total / FLAGS.cross_fold):] + \
                  label_tensor[:int((Round - 1) * sample_total / FLAGS.cross_fold)]
    sample_parts_indexs_train = sample_parts_indexs_tensor[int(Round * sample_total / FLAGS.cross_fold):] + \
                                sample_parts_indexs_tensor[:int((Round - 1) * sample_total / FLAGS.cross_fold)]
    # sample_parts_indexs_train = tf.transpose(sample_parts_indexs_train, [1, 2, 0])
    smaple_adjacent_words_train = sample_adjacent_words_tensor[int(Round * sample_total / FLAGS.cross_fold):] + \
                                  sample_adjacent_words_tensor[:int((Round - 1) * sample_total / FLAGS.cross_fold)]
    sample_flags_train = sample_flags_tensor[int(Round * sample_total / FLAGS.cross_fold):] + \
                         sample_flags_tensor[:int((Round - 1) * sample_total / FLAGS.cross_fold)]

    sample_word_train = np.array(sample_word_train)
    sample_sentence_train = np.array(sample_sentence_train)
    sample_position_train = np.array(sample_position_train)
    label_train = np.array(label_train)
    smaple_adjacent_words_train = np.array(smaple_adjacent_words_train)
    sample_parts_indexs_train = np.array(sample_parts_indexs_train)
    sample_flags_train = np.array(sample_flags_train)
    # print(sample_parts_indexs_train)
    # sample_parts_indexs_train = tf.transpose(sample_parts_indexs_train, [1, 2, 0])

    print("Train/Dev split: {:d}/{:d}".format(len(label_train), len(label_dev)))

    return sample_word_dev, sample_sentence_dev, sample_position_dev, label_dev, smaple_adjacent_words_dev, \
           sample_parts_indexs_dev, sample_flags_dev, sample_word_train, sample_sentence_train, sample_position_train, \
           label_train, smaple_adjacent_words_train, sample_parts_indexs_train, sample_flags_train


def convert_sample(combine_sample):
    """
    Convert Sample
    :param combine_sample:
    :return: Convert sample
    """
    print("Convert data...")
    sample_word_tensor, sample_sentence_tensor, sample_position_tensor, label_tensor, sample_parts_indexs_tensor, \
    sample_adjacent_words_tensor, sample_flags_tensor = zip(*combine_sample)
    print(len(sample_word_tensor))

    sample_word_tensor = np.array(sample_word_tensor)
    sample_sentence_tensor = np.array(sample_sentence_tensor)
    sample_position_tensor = np.array(sample_position_tensor)
    label_tensor = np.array(label_tensor)
    sample_adjacent_words_tensor = np.array(sample_adjacent_words_tensor)
    sample_parts_indexs_tensor = np.array(sample_parts_indexs_tensor)
    sample_flags_tensor = np.array(sample_flags_tensor)
    # print(sample_parts_indexs_train)
    # sample_parts_indexs_train = tf.transpose(sample_parts_indexs_train, [1, 2, 0])

    return sample_word_tensor, sample_sentence_tensor, sample_position_tensor, label_tensor, sample_parts_indexs_tensor, \
           sample_adjacent_words_tensor, sample_flags_tensor


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


if __name__ == '__main__':
    train(100)
