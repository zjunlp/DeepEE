# -*- coding: utf-8 -*-

from DMCNN_trigger_bag import DMCNN_TRIGGER_BAG
import Params
import tensor_trigger_bag
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

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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

# Data preparation
# ====================================================================

# Load data
print(str(Params.cross_fold))
print("Loading data...")
train_word_list_bag, train_sentence_pre_word_id_list_bag, train_word_position_list_bag, train_label_bag, train_adjacent_words_id_bag, train_trigger_parts_index_bag, train_bag_flags = \
    tensor_trigger_bag.get_trigger_bag_tensor(Params.bag_train_wiki_sentence_annotated_with_trigger_path)
test_word_list_bag, test_sentence_pre_word_id_list_bag, test_word_position_list_bag, test_label_bag, test_adjacent_words_id_bag, test_trigger_parts_index_bag, test_bag_flags = \
    tensor_trigger_bag.get_trigger_bag_tensor(Params.bag_test_wiki_sentence_annotated_with_trigger_path)
dev_word_list_bag, dev_sentence_pre_word_id_list_bag, dev_word_position_list_bag, dev_label_bag, dev_adjacent_words_id_bag, dev_trigger_parts_index_bag, dev_bag_flags = \
    tensor_trigger_bag.get_trigger_bag_tensor(Params.bag_dev_wiki_sentence_annotated_with_trigger_path)

train_sample_total = len(train_label_bag)
# print(sample_total)

print("Shuffle sample...")
train_combine_sample = list(zip(train_sentence_pre_word_id_list_bag, train_word_position_list_bag, train_label_bag, train_adjacent_words_id_bag,
                                train_trigger_parts_index_bag, train_bag_flags))
print(len(train_combine_sample))
random.shuffle(train_combine_sample)
print(len(train_combine_sample))

test_combine_sample = list(zip(test_sentence_pre_word_id_list_bag, test_word_position_list_bag, test_label_bag, test_adjacent_words_id_bag,
                               test_trigger_parts_index_bag, test_bag_flags))
print(len(test_combine_sample))
random.shuffle(test_combine_sample)
print(len(test_combine_sample))

dev_combine_sample = list(zip(dev_sentence_pre_word_id_list_bag, dev_word_position_list_bag, dev_label_bag, dev_adjacent_words_id_bag,
                              dev_trigger_parts_index_bag, dev_bag_flags))
print(len(dev_combine_sample))
random.shuffle(dev_combine_sample)
print(len(dev_combine_sample))


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
                gpu_flags='/gpu:3',
                bag_word_length=FLAGS.bag_word_length
            )

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(DMcnn_trigger_bag.loss)
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
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "trigger_bag", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            # Train summaries
            loss_summary = tf.summary.scalar("loss", DMcnn_trigger_bag.loss)
            acc_summary = tf.summary.scalar("accuracy", DMcnn_trigger_bag.accuracy)

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

            # Load data
            sentence_pre_word_id_list_bag_train, word_position_list_bag_train, label_bag_train, \
            adjacent_words_id_bag_train, trigger_parts_index_bag_train, train_bag_flags = split_sample(train_combine_sample)

            sentence_pre_word_id_list_bag_test, word_position_list_bag_test, label_bag_test, \
            adjacent_words_id_bag_test, trigger_parts_index_bag_test, test_bag_flags = split_sample(test_combine_sample)

            sentence_pre_word_id_list_bag_dev, word_position_list_bag_dev, label_bag_dev, \
            adjacent_words_id_bag_dev, trigger_parts_index_bag_dev, dev_bag_flags = split_sample(dev_combine_sample)

            for i in range(Round):
                # results dir
                results_prefix = os.path.join(results_dir, "round-" + str(i + 1) + ".log")
                print("Round train " + str(i % 5 + 1))

                # Train
                batches = batch_iter(
                    list(zip(sentence_pre_word_id_list_bag_train, word_position_list_bag_train, label_bag_train,
                             adjacent_words_id_bag_train, trigger_parts_index_bag_train, train_bag_flags)), FLAGS.batch_size, 1)
                batch_num = 0

                # Write results into file
                with open(results_prefix, 'w', encoding='utf-8') as results_file:
                    for batch in batches:
                        batch_num += 1
                        positive_all_num = 0
                        positive_pre_num = 0
                        positive_pre_correct_num = 0

                        print("batch number %d" % batch_num)
                        sentence_pre_word_id_list_bag_batch, word_position_list_bag_batch, label_bag_batch, \
                        adjacent_words_id_bag_batch, trigger_parts_index_bag_batch, bag_flags_batch = zip(*batch)
                        bag_num_train_batch = len(label_bag_batch)
                        # print(sample_flags_batch)
                        # print('\n')
                        # break

                        sample_label_neg_batch_train = []
                        for i in range(len(label_bag_batch)):
                            sample_neg_flags_temp = [int(0)] * 21
                            sample_neg_flags_temp.append(int(1))
                            sample_label_neg_batch_train.append(sample_neg_flags_temp)
                        sample_label_neg_batch_train = np.array(sample_label_neg_batch_train)

                        scores_lookup_add_train = []
                        scores_lookup_add_train.append(int(0))
                        id_train_num = 0
                        for i in range(len(label_bag_batch) - 1):
                            id_train_num += Params.bag_word_length
                            scores_lookup_add_train.append(int(id_train_num))
                        scores_lookup_add_train = np.array(scores_lookup_add_train)

                        sentence_pre_word_id_list_bag_all_batch = []
                        word_position_list_bag_all_batch = []
                        label_bag_all_batch = []
                        adjacent_words_id_bag_all_batch = []
                        trigger_parts_index_bag_all_batch = []
                        bag_flags_all_batch = []

                        for bag_num in range(len(label_bag_batch)):
                            sentence_pre_word_id_list_bag_all = []
                            word_position_list_bag_all = []
                            # label_bag_all = []
                            adjacent_words_id_bag_all = []
                            trigger_parts_index_bag_all= []
                            bag_flags_all = []
                            for sentence_num in range(len(sentence_pre_word_id_list_bag_batch[bag_num])):
                                sentence_pre_word_id_list_bag_all.extend(sentence_pre_word_id_list_bag_batch[bag_num][sentence_num])
                                word_position_list_bag_all.extend(word_position_list_bag_batch[bag_num][sentence_num])
                                adjacent_words_id_bag_all.extend(adjacent_words_id_bag_batch[bag_num][sentence_num])
                                trigger_parts_index_bag_all.extend(trigger_parts_index_bag_batch[bag_num][sentence_num])
                            # word_num = int(len(sentence_pre_word_id_list_bag_all))

                            # print(len(sentence_pre_word_id_list_bag_all))
                            # print(len(word_position_list_bag_all))
                            # print(len(adjacent_words_id_bag_all))
                            # print(len(trigger_parts_index_bag_all))

                            sentence_pre_word_id_list_bag_all, word_position_list_bag_all, adjacent_words_id_bag_all, trigger_parts_index_bag_all =\
                                cut_bag_word(sentence_pre_word_id_list_bag_all, word_position_list_bag_all, adjacent_words_id_bag_all, trigger_parts_index_bag_all, Params.bag_word_length)

                            # print(len(sentence_pre_word_id_list_bag_all))
                            # print(len(word_position_list_bag_all))
                            # print(len(adjacent_words_id_bag_all))
                            # print(len(trigger_parts_index_bag_all))

                            # sentence_pre_word_id_list_bag_all = np.array(sentence_pre_word_id_list_bag_all)
                            # word_position_list_bag_all = np.array(word_position_list_bag_all)
                            # label_bag_all.append(label_bag_batch[bag_num])
                            # label_bag_all = np.array(label_bag_all)
                            label_bag_all = np.array(label_bag_batch[bag_num])
                            bag_flags_all.append(bag_flags_batch[bag_num])
                            # bag_flags_all = np.array(bag_flags_all)
                            # adjacent_words_id_bag_all = np.array(adjacent_words_id_bag_all)
                            # trigger_parts_index_bag_all = np.array(trigger_parts_index_bag_all)

                            sentence_pre_word_id_list_bag_all_batch.extend(sentence_pre_word_id_list_bag_all)
                            word_position_list_bag_all_batch.extend(word_position_list_bag_all)
                            label_bag_all_batch.append(label_bag_all)
                            adjacent_words_id_bag_all_batch.extend(adjacent_words_id_bag_all)
                            trigger_parts_index_bag_all_batch.extend(trigger_parts_index_bag_all)
                            bag_flags_all_batch.extend(bag_flags_all)

                        sentence_pre_word_id_list_bag_all_batch = np.array(sentence_pre_word_id_list_bag_all_batch)
                        word_position_list_bag_all_batch = np.array(word_position_list_bag_all_batch)
                        label_bag_all_batch = np.array(label_bag_all_batch)
                        adjacent_words_id_bag_all_batch = np.array(adjacent_words_id_bag_all_batch)
                        trigger_parts_index_bag_all_batch = np.array(trigger_parts_index_bag_all_batch)
                        bag_flags_all_batch = np.array(bag_flags_all_batch)

                        # print(len(sentence_pre_word_id_list_bag_all_batch))
                        # print(len(word_position_list_bag_all_batch))
                        # print(len(adjacent_words_id_bag_all_batch))
                        # print(len(trigger_parts_index_bag_all_batch))
                        # print(len(label_bag_all_batch))

                        feed_dict = {DMcnn_trigger_bag.input_sentence: sentence_pre_word_id_list_bag_all_batch,
                                     DMcnn_trigger_bag.input_word_position: word_position_list_bag_all_batch,
                                     DMcnn_trigger_bag.input_role: label_bag_all_batch,
                                     DMcnn_trigger_bag.input_parts_indexs: trigger_parts_index_bag_all_batch,
                                     DMcnn_trigger_bag.input_adjacent_words: adjacent_words_id_bag_all_batch,
                                     DMcnn_trigger_bag.bag_num: bag_num_train_batch,
                                     DMcnn_trigger_bag.input_flags: bag_flags_all_batch,
                                     DMcnn_trigger_bag.input_neg_label: sample_label_neg_batch_train,
                                     DMcnn_trigger_bag.scores_lookup_add: scores_lookup_add_train,
                                     DMcnn_trigger_bag.dropout_keep_prob: FLAGS.dropout_keep_prob
                                     }

                        _, step, summaries, loss, accuracy, pos_pre, pos_pre_all, pos_pre_num, pos_all_num, precision, recall, F1 = sess.run(
                            [train_op, global_step, train_summary_op, DMcnn_trigger_bag.loss, DMcnn_trigger_bag.accuracy,
                             DMcnn_trigger_bag.positive_correct_prediction, DMcnn_trigger_bag.positive_prediction,
                             DMcnn_trigger_bag.positive_correct_prediction_num, DMcnn_trigger_bag.positive_correct_all_num, DMcnn_trigger_bag.precision,
                             DMcnn_trigger_bag.recall_value, DMcnn_trigger_bag.F1_value], feed_dict)
                        time_str = datetime.datetime.now().isoformat()
                        print("train: {}: step {}, loss {:g}, acc {:g}, pre {:g}, recall {:g}, F1 {:g}".format(time_str, step, loss, accuracy, precision, recall, F1))

                        results_file.write("train: {}: step {}, loss {:g}, acc {:g}, pre {:g}, recall {:g}, F1 {:g}".format(time_str, step, loss, accuracy, precision, recall, F1))
                        results_file.write('\n')
                        print('\n')
                        print(pos_pre_all)
                        print(pos_pre_num)
                        print(pos_all_num)
                        print('\n')
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
                                list(zip(sentence_pre_word_id_list_bag_dev, word_position_list_bag_dev, label_bag_dev,
                                        adjacent_words_id_bag_dev, trigger_parts_index_bag_dev, dev_bag_flags)),
                                        FLAGS.batch_size, 0)
                            # ==============================================
                            for batch_dev in batches_dev:
                                batch_dev_num += 1

                                sentence_pre_word_id_list_bag_dev_batch, word_position_list_bag_dev_batch, label_bag_dev_batch, \
                                adjacent_words_id_bag_dev_batch, trigger_parts_index_bag_dev_batch, bag_flag_dev_batch = zip(*batch_dev)
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

                                sentence_pre_word_id_list_bag_all_dev_batch = []
                                word_position_list_bag_all_dev_batch = []
                                label_bag_all_dev_batch = []
                                adjacent_words_id_bag_all_dev_batch = []
                                trigger_parts_index_bag_all_dev_batch = []
                                bag_flags_all_dev_batch = []

                                for bag_num in range(len(label_bag_dev_batch)):
                                    sentence_pre_word_id_list_bag_dev_all = []
                                    word_position_list_bag_dev_all = []
                                    # label_bag_dev_all = []
                                    adjacent_words_id_bag_dev_all = []
                                    trigger_parts_index_bag_dev_all = []
                                    bag_flags_dev_all = []
                                    for sentence_num in range(len(sentence_pre_word_id_list_bag_dev_batch[bag_num])):
                                        sentence_pre_word_id_list_bag_dev_all.extend(sentence_pre_word_id_list_bag_dev_batch[bag_num][sentence_num])
                                        word_position_list_bag_dev_all.extend(word_position_list_bag_dev_batch[bag_num][sentence_num])
                                        adjacent_words_id_bag_dev_all.extend(adjacent_words_id_bag_dev_batch[bag_num][sentence_num])
                                        trigger_parts_index_bag_dev_all.extend(trigger_parts_index_bag_dev_batch[bag_num][sentence_num])
                                    # word_num_dev = int(len(sentence_pre_word_id_list_bag_dev_all))

                                    sentence_pre_word_id_list_bag_dev_all, word_position_list_bag_dev_all, adjacent_words_id_bag_dev_all, trigger_parts_index_bag_dev_all = \
                                        cut_bag_word(sentence_pre_word_id_list_bag_dev_all, word_position_list_bag_dev_all, adjacent_words_id_bag_dev_all, trigger_parts_index_bag_dev_all, Params.bag_word_length)

                                    # sentence_pre_word_id_list_bag_dev_all = np.array(sentence_pre_word_id_list_bag_dev_all)
                                    # word_position_list_bag_dev_all = np.array(word_position_list_bag_dev_all)
                                    # label_bag_dev_all.append(label_bag_dev_batch[bag_num])
                                    # label_bag_dev_all = np.array(label_bag_dev_all)
                                    label_bag_dev_all = np.array(label_bag_dev_batch[bag_num])
                                    bag_flags_dev_all.append(bag_flag_dev_batch[bag_num])
                                    # bag_flags_dev_all = np.array(bag_flags_dev_all)
                                    # adjacent_words_id_bag_dev_all = np.array(adjacent_words_id_bag_dev_all)
                                    # trigger_parts_index_bag_dev_all = np.array(trigger_parts_index_bag_dev_all)

                                    sentence_pre_word_id_list_bag_all_dev_batch.extend(sentence_pre_word_id_list_bag_dev_all)
                                    word_position_list_bag_all_dev_batch.extend(word_position_list_bag_dev_all)
                                    label_bag_all_dev_batch.append(label_bag_dev_all)
                                    adjacent_words_id_bag_all_dev_batch.extend(adjacent_words_id_bag_dev_all)
                                    trigger_parts_index_bag_all_dev_batch.extend(trigger_parts_index_bag_dev_all)
                                    bag_flags_all_dev_batch.extend(bag_flags_dev_all)

                                sentence_pre_word_id_list_bag_all_dev_batch = np.array(sentence_pre_word_id_list_bag_all_dev_batch)
                                word_position_list_bag_all_dev_batch = np.array(word_position_list_bag_all_dev_batch)
                                label_bag_all_dev_batch = np.array(label_bag_all_dev_batch)
                                adjacent_words_id_bag_all_dev_batch = np.array(adjacent_words_id_bag_all_dev_batch)
                                trigger_parts_index_bag_all_dev_batch = np.array(trigger_parts_index_bag_all_dev_batch)
                                bag_flags_all_dev_batch = np.array(bag_flags_all_dev_batch)

                                feed_dict = {DMcnn_trigger_bag.input_sentence: sentence_pre_word_id_list_bag_all_dev_batch,
                                             DMcnn_trigger_bag.input_word_position: word_position_list_bag_all_dev_batch,
                                             DMcnn_trigger_bag.input_role: label_bag_all_dev_batch,
                                             DMcnn_trigger_bag.input_parts_indexs: trigger_parts_index_bag_all_dev_batch,
                                             DMcnn_trigger_bag.input_adjacent_words: adjacent_words_id_bag_all_dev_batch,
                                             DMcnn_trigger_bag.bag_num: bag_num_dev_batch,
                                             DMcnn_trigger_bag.input_flags: bag_flags_all_dev_batch,
                                             DMcnn_trigger_bag.input_neg_label: sample_label_neg_batch_dev,
                                             DMcnn_trigger_bag.scores_lookup_add: scores_lookup_add_dev,
                                             DMcnn_trigger_bag.dropout_keep_prob: FLAGS.dropout_keep_prob}

                                step, summaries, loss, accuracy, pos_pre, pos_pre_all, pos_pre_num, pos_all_num, precision, recall, F1 = sess.run(
                                    [global_step, dev_summary_op, DMcnn_trigger_bag.loss, DMcnn_trigger_bag.accuracy,
                                     DMcnn_trigger_bag.positive_correct_prediction, DMcnn_trigger_bag.positive_prediction,
                                     DMcnn_trigger_bag.positive_correct_prediction_num, DMcnn_trigger_bag.positive_correct_all_num,
                                     DMcnn_trigger_bag.precision, DMcnn_trigger_bag.recall_value, DMcnn_trigger_bag.F1_value], feed_dict)
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
                        if step % FLAGS.checkpoint_pre == 0:
                            path = saver.save(sess, checkpoint_prefix, global_step=step)
                            print("Saved model checkpoint to {}\n".format(path))
                    # ===============================Dev=====================================
                    batch_dev_num = 0
                    loss_dev = 0.0
                    acc_dev = 0.0
                    pos_pre_all_dev = 0
                    pos_pre_num_dev = 0
                    pos_all_num_dev = 0
                    # ============================================
                    batches_dev = batch_iter(
                        list(zip(sentence_pre_word_id_list_bag_dev, word_position_list_bag_dev, label_bag_dev,
                                 adjacent_words_id_bag_dev, trigger_parts_index_bag_dev, dev_bag_flags)),
                        FLAGS.batch_size, 0)
                    # ==============================================
                    for batch_dev in batches_dev:
                        batch_dev_num += 1

                        sentence_pre_word_id_list_bag_dev_batch, word_position_list_bag_dev_batch, label_bag_dev_batch, \
                        adjacent_words_id_bag_dev_batch, trigger_parts_index_bag_dev_batch, bag_flag_dev_batch = zip(*batch_dev)
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

                        sentence_pre_word_id_list_bag_all_dev_batch = []
                        word_position_list_bag_all_dev_batch = []
                        label_bag_all_dev_batch = []
                        adjacent_words_id_bag_all_dev_batch = []
                        trigger_parts_index_bag_all_dev_batch = []
                        bag_flags_all_dev_batch = []

                        for bag_num in range(len(label_bag_dev_batch)):
                            sentence_pre_word_id_list_bag_dev_all = []
                            word_position_list_bag_dev_all = []
                            # label_bag_dev_all = []
                            adjacent_words_id_bag_dev_all = []
                            trigger_parts_index_bag_dev_all = []
                            bag_flags_dev_all = []
                            for sentence_num in range(len(sentence_pre_word_id_list_bag_dev_batch[bag_num])):
                                sentence_pre_word_id_list_bag_dev_all.extend(sentence_pre_word_id_list_bag_dev_batch[bag_num][sentence_num])
                                word_position_list_bag_dev_all.extend(word_position_list_bag_dev_batch[bag_num][sentence_num])
                                adjacent_words_id_bag_dev_all.extend(adjacent_words_id_bag_dev_batch[bag_num][sentence_num])
                                trigger_parts_index_bag_dev_all.extend(trigger_parts_index_bag_dev_batch[bag_num][sentence_num])
                                # word_num_dev = int(len(sentence_pre_word_id_list_bag_dev_all))

                            sentence_pre_word_id_list_bag_dev_all, word_position_list_bag_dev_all, adjacent_words_id_bag_dev_all, trigger_parts_index_bag_dev_all = \
                                cut_bag_word(sentence_pre_word_id_list_bag_dev_all,word_position_list_bag_dev_all, adjacent_words_id_bag_dev_all,trigger_parts_index_bag_dev_all, Params.bag_word_length)

                            # sentence_pre_word_id_list_bag_dev_all = np.array(sentence_pre_word_id_list_bag_dev_all)
                            # word_position_list_bag_dev_all = np.array(word_position_list_bag_dev_all)
                            # label_bag_dev_all.append(label_bag_dev_batch[bag_num])
                            # label_bag_dev_all = np.array(label_bag_dev_all)
                            label_bag_dev_all = np.array(label_bag_dev_batch[bag_num])
                            bag_flags_dev_all.append(bag_flag_dev_batch[bag_num])
                            # bag_flags_dev_all = np.array(bag_flags_dev_all)
                            # adjacent_words_id_bag_dev_all = np.array(adjacent_words_id_bag_dev_all)
                            # trigger_parts_index_bag_dev_all = np.array(trigger_parts_index_bag_dev_all)

                            sentence_pre_word_id_list_bag_all_dev_batch.extend(sentence_pre_word_id_list_bag_dev_all)
                            word_position_list_bag_all_dev_batch.extend(word_position_list_bag_dev_all)
                            label_bag_all_dev_batch.append(label_bag_dev_all)
                            adjacent_words_id_bag_all_dev_batch.extend(adjacent_words_id_bag_dev_all)
                            trigger_parts_index_bag_all_dev_batch.extend(trigger_parts_index_bag_dev_all)
                            bag_flags_all_dev_batch.extend(bag_flags_dev_all)

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

                        step, summaries, loss, accuracy, pos_pre, pos_pre_all, pos_pre_num, pos_all_num, precision, recall, F1 = sess.run(
                            [global_step, dev_summary_op, DMcnn_trigger_bag.loss, DMcnn_trigger_bag.accuracy,
                             DMcnn_trigger_bag.positive_correct_prediction,
                             DMcnn_trigger_bag.positive_prediction,
                             DMcnn_trigger_bag.positive_correct_prediction_num,
                             DMcnn_trigger_bag.positive_correct_all_num,
                             DMcnn_trigger_bag.precision, DMcnn_trigger_bag.recall_value,
                             DMcnn_trigger_bag.F1_value], feed_dict)
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
                    # ===============================Test====================================
                    batch_test_num = 0
                    loss_test = 0.0
                    acc_test = 0.0
                    pos_pre_all_test = 0
                    pos_pre_num_test = 0
                    pos_all_num_test = 0
                    # ============================================
                    batches_test = batch_iter(
                        list(zip(sentence_pre_word_id_list_bag_test, word_position_list_bag_test, label_bag_test,
                                 adjacent_words_id_bag_test, trigger_parts_index_bag_test, test_bag_flags)),
                        FLAGS.batch_size, 0)
                    # ==============================================
                    for batch_test in batches_test:
                        batch_test_num += 1

                        sentence_pre_word_id_list_bag_test_batch, word_position_list_bag_test_batch, label_bag_test_batch, \
                        adjacent_words_id_bag_test_batch, trigger_parts_index_bag_test_batch, bag_flag_test_batch = zip(*batch_test)
                        bag_num_test_batch = len(label_bag_test_batch)

                        sample_label_neg_batch_test = []
                        for i in range(len(label_bag_test_batch)):
                            sample_neg_flags_temp = [int(0)] * 21
                            sample_neg_flags_temp.append(int(1))
                            sample_label_neg_batch_test.append(sample_neg_flags_temp)
                        sample_label_neg_batch_test = np.array(sample_label_neg_batch_test)

                        scores_lookup_add_test = []
                        scores_lookup_add_test.append(int(0))
                        id_test_num = 0
                        for i in range(len(label_bag_test_batch) - 1):
                            id_test_num += Params.bag_word_length
                            scores_lookup_add_test.append(int(id_test_num))
                        scores_lookup_add_test = np.array(scores_lookup_add_test)

                        sentence_pre_word_id_list_bag_all_test_batch = []
                        word_position_list_bag_all_test_batch = []
                        label_bag_all_test_batch = []
                        adjacent_words_id_bag_all_test_batch = []
                        trigger_parts_index_bag_all_test_batch = []
                        bag_flags_all_test_batch = []

                        for bag_num in range(len(label_bag_test_batch)):
                            sentence_pre_word_id_list_bag_test_all = []
                            word_position_list_bag_test_all = []
                            # label_bag_test_all = []
                            adjacent_words_id_bag_test_all = []
                            trigger_parts_index_bag_test_all = []
                            bag_flags_test_all = []
                            for sentence_num in range(len(sentence_pre_word_id_list_bag_test_batch[bag_num])):
                                sentence_pre_word_id_list_bag_test_all.extend(sentence_pre_word_id_list_bag_test_batch[bag_num][sentence_num])
                                word_position_list_bag_test_all.extend(word_position_list_bag_test_batch[bag_num][sentence_num])
                                adjacent_words_id_bag_test_all.extend(adjacent_words_id_bag_test_batch[bag_num][sentence_num])
                                trigger_parts_index_bag_test_all.extend(trigger_parts_index_bag_test_batch[bag_num][sentence_num])
                                # word_num_test = int(len(sentence_pre_word_id_list_bag_test_all))

                            sentence_pre_word_id_list_bag_test_all, word_position_list_bag_test_all, adjacent_words_id_bag_test_all, trigger_parts_index_bag_test_all = \
                                cut_bag_word(sentence_pre_word_id_list_bag_test_all, word_position_list_bag_test_all, adjacent_words_id_bag_test_all, trigger_parts_index_bag_test_all, Params.bag_word_length)

                            # sentence_pre_word_id_list_bag_test_all = np.array(sentence_pre_word_id_list_bag_test_all)
                            # word_position_list_bag_test_all = np.array(word_position_list_bag_test_all)
                            # label_bag_test_all.append(label_bag_test_batch[bag_num])
                            # label_bag_test_all = np.array(label_bag_test_all)
                            label_bag_test_all = np.array(label_bag_test_batch[bag_num])
                            bag_flags_test_all.append(bag_flag_test_batch[bag_num])
                            # bag_flags_test_all = np.array(bag_flags_test_all)
                            # adjacent_words_id_bag_test_all = np.array(adjacent_words_id_bag_test_all)
                            # trigger_parts_index_bag_test_all = np.array(trigger_parts_index_bag_test_all)

                            sentence_pre_word_id_list_bag_all_test_batch.extend(sentence_pre_word_id_list_bag_test_all)
                            word_position_list_bag_all_test_batch.extend(word_position_list_bag_test_all)
                            label_bag_all_test_batch.append(label_bag_test_all)
                            adjacent_words_id_bag_all_test_batch.extend(adjacent_words_id_bag_test_all)
                            trigger_parts_index_bag_all_test_batch.extend(trigger_parts_index_bag_test_all)
                            bag_flags_all_test_batch.extend(bag_flags_test_all)

                        sentence_pre_word_id_list_bag_all_test_batch = np.array(sentence_pre_word_id_list_bag_all_test_batch)
                        word_position_list_bag_all_test_batch = np.array(word_position_list_bag_all_test_batch)
                        label_bag_all_test_batch = np.array(label_bag_all_test_batch)
                        adjacent_words_id_bag_all_test_batch = np.array(adjacent_words_id_bag_all_test_batch)
                        trigger_parts_index_bag_all_test_batch = np.array(trigger_parts_index_bag_all_test_batch)
                        bag_flags_all_test_batch = np.array(bag_flags_all_test_batch)

                        feed_dict = {
                            DMcnn_trigger_bag.input_sentence: sentence_pre_word_id_list_bag_all_test_batch,
                            DMcnn_trigger_bag.input_word_position: word_position_list_bag_all_test_batch,
                            DMcnn_trigger_bag.input_role: label_bag_all_test_batch,
                            DMcnn_trigger_bag.input_parts_indexs: trigger_parts_index_bag_all_test_batch,
                            DMcnn_trigger_bag.input_adjacent_words: adjacent_words_id_bag_all_test_batch,
                            DMcnn_trigger_bag.bag_num: bag_num_test_batch,
                            DMcnn_trigger_bag.input_flags: bag_flags_all_test_batch,
                            DMcnn_trigger_bag.scores_lookup_add: scores_lookup_add_test,
                            DMcnn_trigger_bag.input_neg_label: sample_label_neg_batch_test,
                            DMcnn_trigger_bag.dropout_keep_prob: FLAGS.dropout_keep_prob}

                        step, summaries, loss, accuracy, pos_pre, pos_pre_all, pos_pre_num, pos_all_num, precision, recall, F1 = sess.run(
                            [global_step, dev_summary_op, DMcnn_trigger_bag.loss, DMcnn_trigger_bag.accuracy,
                             DMcnn_trigger_bag.positive_correct_prediction,
                             DMcnn_trigger_bag.positive_prediction,
                             DMcnn_trigger_bag.positive_correct_prediction_num,
                             DMcnn_trigger_bag.positive_correct_all_num,
                             DMcnn_trigger_bag.precision, DMcnn_trigger_bag.recall_value,
                             DMcnn_trigger_bag.F1_value], feed_dict)
                        time_str = datetime.datetime.now().isoformat()
                        loss_test = loss_test + loss
                        acc_test = acc_test + accuracy
                        pos_pre_all_test = pos_pre_all_test + pos_pre_all
                        pos_pre_num_test = pos_pre_num_test + pos_pre_num
                        pos_all_num_test = pos_all_num_test + pos_all_num

                    if dev_summary_writer:
                        dev_summary_writer.add_summary(summaries, step)
                    print("")

                    print(batch_test_num)
                    loss = loss_test / batch_test_num
                    accuracy = acc_test / batch_test_num
                    precision = pos_pre_num_test / pos_pre_all_test
                    recall = pos_pre_num_test / pos_all_num_test
                    F1 = (2 * precision * recall) / (precision + recall)

                    print("Test: {}: step {}, loss {:g}, acc {:g}, pre {:g}, recall {:g}, F1 {:g}".format(time_str, step, loss, accuracy, precision, recall, F1))
                    results_file.write("Dev: {}: step {}, loss {:g}, acc {:g}, pre {:g}, recall {:g}, F1 {:g}".format(time_str, step, loss, accuracy, precision, recall, F1))
                    results_file.write('\n')
                    results_file.write('\n')
                    print('\n')
                    print(pos_pre_all_test)
                    print(pos_pre_num_test)
                    print(pos_all_num_test)
                    print('\n')
                results_file.close()


def cut_bag_word(sentence_pre_word_id_list_bag_all, word_position_list_bag_all, adjacent_words_id_bag_all, trigger_parts_index_bag_all, bag_word_length):
    """
    Cut bag word
    :param sentence_pre_word_id_list_bag_all:
    :param word_position_list_bag_all:
    :param adjacent_words_id_bag_all:
    :param trigger_parts_index_bag_all:
    :return:
    """
    sentence_pre_word_id_list_bag_all_cut = []
    word_position_list_bag_all_cut = []
    adjacent_words_id_bag_all_cut = []
    trigger_parts_index_bag_all_cut = []
    if len(sentence_pre_word_id_list_bag_all) <= bag_word_length:
        num = bag_word_length - len(sentence_pre_word_id_list_bag_all)

        sentence_pre_word_id_list_bag_all_cut = sentence_pre_word_id_list_bag_all
        word_position_list_bag_all_cut = word_position_list_bag_all
        adjacent_words_id_bag_all_cut = adjacent_words_id_bag_all
        trigger_parts_index_bag_all_cut = trigger_parts_index_bag_all

        for i in range(num):
            sentence_pre_word_id_list_bag_all_cut.append(sentence_pre_word_id_list_bag_all[0])
            word_position_list_bag_all_cut.append(word_position_list_bag_all[0])
            adjacent_words_id_bag_all_cut.append(adjacent_words_id_bag_all[0])
            trigger_parts_index_bag_all_cut.append(trigger_parts_index_bag_all[0])
    else:
        for i in range(bag_word_length):
            sentence_pre_word_id_list_bag_all_cut.append(sentence_pre_word_id_list_bag_all[i])
            word_position_list_bag_all_cut.append(word_position_list_bag_all[i])
            adjacent_words_id_bag_all_cut.append(adjacent_words_id_bag_all[i])
            trigger_parts_index_bag_all_cut.append(trigger_parts_index_bag_all[i])

    return sentence_pre_word_id_list_bag_all_cut, word_position_list_bag_all_cut, adjacent_words_id_bag_all_cut, trigger_parts_index_bag_all_cut


def split_sample(combine_sample):
    """
    Split Sample
    :param combine_sample:
    :return: split sample
    """
    sentence_pre_word_id_list_bag, word_position_list_bag, label_bag, adjacent_words_id_bag, trigger_parts_index_bag, bag_flags = zip(*combine_sample)
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

    return sentence_pre_word_id_list_bag, word_position_list_bag, label_bag, adjacent_words_id_bag, trigger_parts_index_bag, bag_flags


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
    train(80)