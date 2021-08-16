# -*- coding: utf-8 -*-

"""
DMCNN model file
Author: ZUO XINYU
Date: 07/23/7017
"""

import tensorflow as tf
import math


class DMCNN(object):
    """
    A DMCNN(Dynamic Multi-Pooling Convolutional Neural) model.
    """

    def __init__(self, sequence_length, num_role, vocab_size, embedding_size, position_size,
                 position_embedding_size, event_type_num, event_type_embedding_size, filter_size, num_filters,
                 trigger_flags, gpu_flags, l2_reg_lambda=0.0):
        # initializer
        initializer = tf.contrib.layers.xavier_initializer()

        # Placeholders for input sentence, input argument role, dropout
        self.input_sentence = tf.placeholder(tf.int32, [None, sequence_length], name='input_sentence')
        # self.input_word = tf.placeholder(tf.int32, [None, 1], name='input_word')
        self.input_type = tf.placeholder(tf.int32, [None, sequence_length], name='input_type')
        self.input_word_position = tf.placeholder(tf.int32, [None, sequence_length], name='input_word_position')
        self.input_trigger_position = tf.placeholder(tf.int32, name='input_trigger_position')
        self.input_parts_indexs = tf.placeholder(tf.float32, [None, 2, sequence_length], name='inout_parts_indexs')
        self.input_role = tf.placeholder(tf.float32, [None, num_role], name='input_role')
        self.input_adjacent_words = tf.placeholder(tf.int32, [None, 3], name='input_role')
        self.input_flags = tf.placeholder(tf.int32, [None], name='input_flags')
        self.input_neg_label = tf.placeholder(tf.int32, [None, num_role], name='input_neg_label')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # keeping track of l2 regularization loss
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device(gpu_flags), tf.name_scope('embedding'):
            self.word_lookup = tf.get_variable(name='word_embedding', shape=[vocab_size, embedding_size],
                                               initializer=initializer)
            # self.W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]), trainable=False, name='W')
            # self.word_lookup_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
            # self.word_lookup = self.W.assign(self.word_lookup_placeholder)
            self.PF_lookup = tf.get_variable(name='position_embedding', shape=[position_size, position_embedding_size],
                                             initializer=initializer)
            self.EF_lookup = tf.get_variable(name='type_embedding', shape=[event_type_num, event_type_embedding_size],
                                             initializer=initializer)

            with tf.variable_scope('CWF'):
                self.CWF = tf.nn.embedding_lookup(self.word_lookup, self.input_sentence)
            # print(self.CWF)

            # Lexical feature representation
            with tf.variable_scope('Lexical_FR'):
                self.lexical_feature = tf.nn.embedding_lookup(self.word_lookup, self.input_adjacent_words)
                self.lexical_feature = tf.reshape(self.lexical_feature, [-1, 3 * embedding_size])

            with tf.variable_scope('PF'):
                self.PF = tf.nn.embedding_lookup(self.PF_lookup, self.input_word_position)
                # self.PF = tf.reshape(self.PF, [-1])
            # print(self.PF)

            with tf.variable_scope('EF'):
                if trigger_flags == 1:
                    self.EF = tf.nn.embedding_lookup(self.EF_lookup, self.input_type)
                    # self.EF = tf.reshape(self.PF, [-1])
                    # print(self.EF)

            with tf.variable_scope('embedding_all'):
                if trigger_flags == 1:
                    self.embedding_all = tf.concat([self.CWF, self.PF, self.PF, self.EF], 2)
                else:
                    self.embedding_all = tf.concat([self.CWF, self.PF, self.PF], 2)
                self.embedding_all_expand = tf.expand_dims(self.embedding_all, -1)
                # print(self.embedding_all)

        # Create convolution layer
        with tf.device(gpu_flags), tf.name_scope('Convolution-%s' % filter_size):
            # =======================================================================================
            head_inds, tail_inds = tf.unstack(tf.transpose(self.input_parts_indexs, [1, 2, 0]))
            self.embedding_all_expand = tf.transpose(self.embedding_all_expand, [2, 3, 1, 0])
            embedding_all_expand_part1 = self.embedding_all_expand * head_inds
            embedding_all_expand_part2 = self.embedding_all_expand * tail_inds

            embedding_all_expand_part1 = tf.transpose(embedding_all_expand_part1, [3, 2, 0, 1])
            embedding_all_expand_part2 = tf.transpose(embedding_all_expand_part2, [3, 2, 0, 1])
            # =======================================================================================

            # filter_embedding_size = embedding_size + 2 * position_embedding_size + event_type_embedding_size
            filter_embedding_size = embedding_size + 2 * position_embedding_size
            filter_shape = [filter_size, filter_embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')

            conv_1 = tf.nn.conv2d(
                embedding_all_expand_part1,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name='conv_1'
            )
            # print(conv_1)

            conv_2 = tf.nn.conv2d(
                embedding_all_expand_part2,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name='conv_2'
            )
            # print(conv_2)

            # Apply a nonlinearity
            h1 = tf.nn.relu(tf.nn.bias_add(conv_1, b), name='relu_1')
            h2 = tf.nn.relu(tf.nn.bias_add(conv_2, b), name='relu_2')

        # Multi-Max pooling
        with tf.device(gpu_flags), tf.name_scope('Multi-pooling%s' % filter_size):
            pooling_output = []
            # Subsection by trigger and argument
            # ========================================argument classification======================

        #     pool_num = 3
            # part_one = tf.ones([1, min(self.input_word_position, self.input_trigger_position)])
            # part_two = tf.ones([1, max(self.input_trigger_position, self.input_word_position) -
            #                     min(self.input_word_position, self.input_trigger_position)])
            # part_three = tf.ones([1, sequence_length - min(self.input_word_position,
            #                                                self.input_trigger_position)])
            # head_inds, mid_inds, tail_inds = tf.unstack(self.input_parts_indexs)
            # h = tf.transpose(h, [2, 3, 1, 0])
            # h1 = h * head_inds
            # h2 = h * mid_inds
            # h3 = h * tail_inds
            #
            # h1 = tf.transpose(h1, [3, 2, 0, 1])
            # h2 = tf.transpose(h2, [3, 2, 0, 1])
            # h3 = tf.transpose(h3, [3, 2, 0, 1])
            #=====================================================================================================
            #
            # pooled_1 = tf.nn.max_pool(
            #     h1,
            #     ksize=[1, sequence_length - filter_size + 1, 1, 1],
            #     strides=[1, sequence_length - filter_size + 1, 1, 1],
            #     padding='SAME',
            #     name='pool_1'
            # )
            #
            # pooled_2 = tf.nn.max_pool(
            #     h2,
            #     ksize=[1, sequence_length - filter_size + 1, 1, 1],
            #     strides=[1, sequence_length - filter_size + 1, 1, 1],
            #     padding='SAME',
            #     name='pool_2'
            # )
            #
            # pooled_3 = tf.nn.max_pool(
            #     h3,
            #     ksize=[1, sequence_length - filter_size + 1, 1, 1],
            #     strides=[1, sequence_length - filter_size + 1, 1, 1],
            #     padding='SAME',
            #     name='pool_3'
            # )
            #
            # pooling_output.append(pooled_1)
            # pooling_output.append(pooled_2)
            # pooling_output.append(pooled_3)
            #

            # =====================================trigger classification========================
            pool_num = 2
            # part_one = tf.ones([1, self.input_word_position])
            # part_two = tf.ones([1, sequence_length - self.input_word_position])
            # head_inds, tail_inds = tf.unstack(tf.transpose(self.input_parts_indexs, [1, 2, 0]))
            # h = tf.transpose(h, [2, 3, 1, 0])
            # h1 = h * head_inds
            # h2 = h * tail_inds
            #
            # h1 = tf.transpose(h1, [3, 2, 0, 1])
            # h2 = tf.transpose(h2, [3, 2, 0, 1])

            # =========================================================================================

            pooled_1 = tf.nn.max_pool(
                h1,
                ksize=[1, sequence_length - filter_size + 1, 1, 1],
                strides=[1, sequence_length - filter_size + 1, 1, 1],
                padding='VALID',
                name='pool_1'
            )

            pooled_2 = tf.nn.max_pool(
                h2,
                ksize=[1, sequence_length - filter_size + 1, 1, 1],
                strides=[1, sequence_length - filter_size + 1, 1, 1],
                padding='VALID',
                name='pool_2'
            )
            # print(pooled_1)
            # print(pooled_2)
            pooling_output.append(pooled_1)
            pooling_output.append(pooled_2)

            # Combine all the pooled features
            num_pool_total = num_filters * pool_num
            self.h_pool = tf.concat(pooling_output, 3)
            # print(self.h_pool)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_pool_total])
            # print(self.h_pool_flat)

        # Combine pooling and lexical & dropout
        with tf.device(gpu_flags), tf.name_scope('Combine_output_and_dropout'):
            # print(self.lexical_feature)
            # self.lexical_feature_flat = tf.reshape(self.lexical_feature, [-1, num_pool_total])
            self.combine_output = tf.concat([self.lexical_feature, self.h_pool_flat], 1)
            # print(self.combine_output)
            self.combine_output_drop = tf.nn.dropout(self.combine_output, self.dropout_keep_prob)

        # Final output use softmax
        with tf.device(gpu_flags), tf.name_scope('output'):
            W = tf.get_variable(
                name='W',
                shape=[num_pool_total + 3 * embedding_size, num_role],
                initializer=initializer
            )
            b = tf.Variable(tf.constant(0.1, shape=[num_role]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.combine_output_drop, W, b, name='scores')
            self.prediction = tf.arg_max(self.scores, 1, name='prediction')

        # CalculationMean cross-entropy loss
        with tf.device(gpu_flags), tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_role)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.device(gpu_flags), tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.prediction, tf.arg_max(self.input_role, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'), name='accuracy')

        # Precision and Recall value
        with tf.device(gpu_flags), tf.name_scope('Precision_Recall'):
            size = tf.size(self.input_role)
            input_role_positive = tf.slice(self.input_role, [0, 0], [tf.to_int32(tf.div(size, 22)), 21])
            self.positive_correct_prediction = tf.equal(self.prediction, tf.arg_max(input_role_positive, 1))
            self.positive_correct_prediction_num = tf.reduce_sum(tf.cast(self.positive_correct_prediction, 'float'))
            self.positive_correct_all_num = tf.reduce_sum(self.input_flags)
            self.recall_value = tf.div(tf.to_float(self.positive_correct_prediction_num),
                                    tf.to_float(self.positive_correct_all_num + 1))

            negative_prediction = tf.equal(self.prediction, tf.arg_max(self.input_neg_label, 1))
            self.positive_prediction = tf.subtract(tf.to_float(tf.div(size, 22)), tf.reduce_sum(tf.cast(negative_prediction, 'float')))
            self.precision = tf.div(tf.to_float(self.positive_correct_prediction_num), tf.to_float(self.positive_prediction))

        # F value
        with tf.device(gpu_flags), tf.name_scope('F1'):
            self.F1_value = tf.div(
                tf.multiply(tf.multiply(tf.to_float(self.precision), tf.to_float(self.recall_value)), tf.to_float(2.0)),
                tf.add(tf.to_float(self.precision), tf.to_float(self.recall_value)))

