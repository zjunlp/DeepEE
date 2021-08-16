import numpy as np
import torch
from torch.utils import data
import json

from consts import NONE, PAD, TRIGGERS, ENTITIES, POSTAGS, MAXLEN, wordemb_file
from utils import build_vocab, load_embedding

# init vocab
all_triggers, trigger2idx, idx2trigger = build_vocab(TRIGGERS, BIO_tagging=False)
all_entities, entity2idx, idx2entity = build_vocab(ENTITIES)
all_postags, postag2idx, idx2postag = build_vocab(POSTAGS, BIO_tagging=False)
word2id, wordemb = load_embedding(wordemb_file)


class ACE2005Dataset(data.Dataset):
    def __init__(self, fpath):
        self.sent_li, self.triggers_li, self.entities_li, self.postags_li, self.adj = [], [], [], [], []

        with open(fpath, 'r') as f:
            data = json.load(f)
            for item in data:
                words = item['words'][:MAXLEN]
                if len(words) < 5:
                    continue
                triggers = [NONE] * len(words)
                entities = [[NONE] for _ in range(len(words))]
                postags = item['pos-tags'][:MAXLEN]
                adjpos, adjv = generateAdjMatrix(item["stanford-colcc"],len(words))

                for entity_mention in item['golden-entity-mentions']:
                    for i in range(entity_mention['start'], entity_mention['end']):
                        entity_type = entity_mention['entity-type']
                        if i < MAXLEN:
                            if i == entity_mention['start']:
                                entity_type = 'B-{}'.format(entity_type)
                            else:
                                entity_type = 'I-{}'.format(entity_type)
                            if len(entities[i]) == 1 and entities[i][0] == NONE:
                                entities[i][0] = entity_type
                            else:
                                entities[i].append(entity_type)

                for event_mention in item['golden-event-mentions']:
                    for i in range(event_mention['trigger']['start'], event_mention['trigger']['end']):
                        if i < MAXLEN:
                            triggers[i] = event_mention['event_type']
                
                self.sent_li.append(words)
                self.triggers_li.append(triggers)
                self.entities_li.append(entities)
                self.postags_li.append(postags)
                self.adj.append([adjpos, adjv])


    def __len__(self):
        return len(self.sent_li)

    def __getitem__(self, idx):
        sentence_li, triggers_li, entities_li, postags_li, adj_li = self.sent_li[idx], self.triggers_li[idx], self.entities_li[idx], self.postags_li[idx], self.adj[idx]
        
        tokens = [word2id[w] if w in word2id else 1 for w in sentence_li]
        triggers = [trigger2idx[t] for t in triggers_li]
        postags = [postag2idx[p] for p in postags_li]
        entities = [[entity2idx[e] for e in ent] for ent in entities_li]
        seqlen = len(tokens)

        return tokens, triggers, entities, postags, adj_li, seqlen, sentence_li, triggers_li

    def get_samples_weight(self):
        samples_weight = []
        for triggers in self.triggers_li:
            not_none = False
            for trigger in triggers:
                if trigger != NONE:
                    not_none = True
                    break
            if not_none:
                samples_weight.append(5.0)
            else:
                samples_weight.append(1.0)
        return np.array(samples_weight)


def pad(batch):
    tokens_2d, triggers_2d, entities_3d, postags_2d, adj, seqlen_1d, words, triggers = list(map(list, zip(*batch)))
    maxlen = np.array(seqlen_1d).max()

    for i in range(len(tokens_2d)):
        tokens_2d[i] = tokens_2d[i] + [0] * (maxlen - len(tokens_2d[i]))
        triggers_2d[i] = triggers_2d[i] + [trigger2idx[PAD]] * (maxlen - len(triggers_2d[i]))
        entities_3d[i] = entities_3d[i] + [[entity2idx[PAD]] for _ in range(maxlen - len(entities_3d[i]))]
        postags_2d[i] = postags_2d[i] + [postag2idx[PAD]] * (maxlen - len(postags_2d[i]))


    return tokens_2d, triggers_2d, entities_3d, postags_2d, adj, seqlen_1d, words, triggers


# Reused from https://github.com/lx865712528/EMNLP2018-JMEE
def generateAdjMatrix(edgeJsonList, Len):
    sparseAdjMatrixPos = [[], [], []]
    sparseAdjMatrixValues = []

    def addedge(type_, from_, to_, value_):
        sparseAdjMatrixPos[0].append(type_)
        sparseAdjMatrixPos[1].append(from_)
        sparseAdjMatrixPos[2].append(to_)
        sparseAdjMatrixValues.append(value_)

    for edgeJson in edgeJsonList:
        ss = edgeJson.split("/")
        fromIndex = int(ss[-1].split("=")[-1])
        toIndex = int(ss[-2].split("=")[-1])
        etype = ss[0].split(":")[0]
        if etype == "root" or fromIndex == -1 or toIndex == -1 or fromIndex >= MAXLEN or toIndex >= MAXLEN:
            continue
        addedge(0, fromIndex, toIndex, 1.0)
        addedge(1, toIndex, fromIndex, 1.0)

    for i in range(Len):
        addedge(2, i, i, 1.0)

    return sparseAdjMatrixPos, sparseAdjMatrixValues