import sys
import torch
import pickle
import random
import numpy as np

from tqdm import tqdm
from allennlp.common.util import pad_sequence_to_length
from allennlp.nn.util import get_mask_from_sequence_lengths

class Dataset(object):
    def __init__(self, batch_size, seq_len, dataset):
        super(Dataset, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.construct_index(dataset)

    def construct_index(self, dataset):
        self.dataset = dataset
        self.index_length = len(dataset)
        self.shuffle_list = list(range(0, self.index_length))


    def shuffle(self):
        random.shuffle(self.shuffle_list)

    def get_tqdm(self, device, shuffle=True):
        return tqdm(self.reader(device, shuffle), mininterval=2, total=self.index_length // self.batch_size, leave=False, file=sys.stdout, ncols=80)

    def reader(self, device, shuffle):
        cur_idx = 0
        while cur_idx < self.index_length:
            end_index = min(cur_idx + self.batch_size, self.index_length)
            batch = [self.dataset[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            cur_idx = end_index
            yield self.batchify(batch, device)
        if shuffle:
            self.shuffle()

    def batchify(self, batch, device):

        input_ids = list()
        input_mask = list()
        segment_ids = list()
        token_to_orig_map = list()
        start_position = list()
        end_position = list()
        example = list()

        for data in batch:
            input_ids.append(data[0])
            input_mask.append(data[1])
            segment_ids.append(data[2])
            token_to_orig_map.append(data[3])
            start_position.append(data[4])
            end_position.append(data[5])
            example.append(data[6])

        return [torch.LongTensor(input_ids).to(device), 
                torch.LongTensor(input_mask).to(device),
                torch.LongTensor(segment_ids).to(device),
                torch.LongTensor(start_position).to(device),
                torch.LongTensor(end_position).to(device),
                token_to_orig_map, example]
