import torch
import torch.nn as nn
from torch.nn import functional as F
from consts import NONE, PAD, WORD_DIM, MAXLEN


def build_vocab(labels, BIO_tagging=True):
    all_labels = [PAD, NONE]
    for label in labels:
        if BIO_tagging:
            all_labels.append('B-{}'.format(label))
            all_labels.append('I-{}'.format(label))
        else:
            all_labels.append(label)
    label2idx = {tag: idx for idx, tag in enumerate(all_labels)}
    idx2label = {idx: tag for idx, tag in enumerate(all_labels)}

    return all_labels, label2idx, idx2label

def load_embedding(wordemb_path):
    word2idx = {}
    wordemb = []
    with open(wordemb_path,'r',encoding='utf-8') as f:
        for line in f:
            splt = line.split()
            assert len(splt)==WORD_DIM+1
            vector = list(map(float, splt[-WORD_DIM:]))
            word = splt[0]
            word2idx[word] = len(word2idx)
            wordemb.append(vector)
    return word2idx, torch.DoubleTensor(wordemb)

def calc_metric(y_true, y_pred):
    """
    :param y_true: [(tuple), ...]
    :param y_pred: [(tuple), ...]
    :return:
    """
    num_proposed = len(y_pred)
    num_gold = len(y_true)

    y_true_set = set(y_true)
    num_correct = 0
    for item in y_pred:
        if item in y_true_set:
            num_correct += 1

    print('proposed: {}\tcorrect: {}\tgold: {}'.format(num_proposed, num_correct, num_gold))

    if num_proposed != 0:
        precision = num_correct / num_proposed
    else:
        precision = 1.0

    if num_gold != 0:
        recall = num_correct / num_gold
    else:
        recall = 1.0

    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1


def find_triggers(labels):

    result = []

    for i in range(len(labels)):
        if labels[i] != NONE:
            result.append([i, i + 1, labels[i]])

    return [tuple(item) for item in result]


# Reused from https://github.com/lx865712528/EMNLP2018-JMEE
class EmbeddingLayer(nn.Module):
    def __init__(self, embedding_size=None, embedding_matrix=None,
                 fine_tune=True, dropout=0.3,
                 padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False,
                 device=torch.device("cpu")):
        super(EmbeddingLayer, self).__init__()
        if embedding_matrix is not None:
            embedding_size = embedding_matrix.size()
        else:
            embedding_matrix = torch.nn.init.uniform_(torch.FloatTensor(embedding_size[0], embedding_size[1]),
                                                      a=-0.15,
                                                      b=0.15)
        assert (embedding_size is not None)
        assert (embedding_matrix is not None)
        # Config copying
        self.matrix = nn.Embedding(num_embeddings=embedding_size[0],
                                   embedding_dim=embedding_size[1],
                                   padding_idx=padding_idx,
                                   max_norm=max_norm,
                                   norm_type=norm_type,
                                   scale_grad_by_freq=scale_grad_by_freq,
                                   sparse=sparse)
        self.matrix.weight.data.copy_(embedding_matrix)
        self.matrix.weight.requires_grad = fine_tune
        self.dropout = dropout if type(dropout) == float and -1e-7 < dropout < 1 + 1e-7 else None

        self.device = device
        self.to(device)

    def forward(self, x):
        if self.dropout is not None:
            return F.dropout(self.matrix(x), p=self.dropout, training=self.training)
        else:
            return self.matrix(x)

class MultiLabelEmbeddingLayer(nn.Module):
    def __init__(self,
                 num_embeddings=None, embedding_dim=None,
                 dropout=0.5, padding_idx=0,
                 max_norm=None, norm_type=2,
                 device=torch.device("cpu")):
        super(MultiLabelEmbeddingLayer, self).__init__()

        self.matrix = nn.Embedding(num_embeddings=num_embeddings,
                                  embedding_dim=embedding_dim,
                                  padding_idx=padding_idx,
                                  max_norm=max_norm,
                                  norm_type=norm_type)
        self.dropout = dropout
        self.device = device
        self.to(device)

    def forward(self, x):
        batch_size = len(x)
        seq_len = len(x[0])
        x = [self.matrix(torch.LongTensor(x[i][j]).to(self.device)).sum(0)
             for i in range(batch_size)
             for j in range(seq_len)]
        x = torch.stack(x).view(batch_size, seq_len, -1)

        if self.dropout is not None:
            return F.dropout(x, p=self.dropout, training=self.training)
        else:
            return x

def get_positions(BATCH_SIZE, SEQ_LEN):
    positions = [[j+MAXLEN for j in range(-i, SEQ_LEN - i)] for i in range(SEQ_LEN)]  # list [SEQ_LEN, SEQ_LEN]
    positions = [torch.LongTensor(position) for position in positions]  # list of tensors [SEQ_LEN]
    positions = [torch.cat([position] * BATCH_SIZE).resize_(BATCH_SIZE, position.size(0))
                 for position in positions]  # list of tensors [BATCH_SIZE, SEQ_LEN]
    return positions
