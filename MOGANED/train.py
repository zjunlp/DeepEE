import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import consts
from model import Net
from data_load import ACE2005Dataset, pad, all_triggers, all_entities, all_postags, word2id, wordemb
from eval import eval


def train(model, iterator, optimizer, criterion):
    model.train()
    for i, batch in enumerate(iterator):
        tokens_2d, triggers_2d, entities_3d, postags_2d, adj, seqlen_1d, words, triggers = batch
        optimizer.zero_grad()
        trigger_logits, trigger_hat_2d = model.predict_triggers(tokens_2d=tokens_2d, entities_3d=entities_3d,
                                    postags_2d=postags_2d, seqlen_1d=seqlen_1d, adjm=adj)

        triggers_y_2d = torch.LongTensor(triggers_2d).to(model.device)
        trigger_logits = trigger_logits.view(-1, trigger_logits.shape[-1])
        trigger_loss = criterion(trigger_logits, triggers_y_2d.view(-1))

        loss = trigger_loss
        loss.backward()

        optimizer.step()
        if i % 40 == 0:  # monitoring
            print("step: {}, loss: {}".format(i, loss.item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=consts.batch_size)
    parser.add_argument("--lr", type=float, default=consts.lr)
    parser.add_argument("--n_epochs", type=int, default=consts.n_epochs)
    parser.add_argument("--logdir", type=str, default="output/logdir2")
    parser.add_argument("--trainset", type=str, default=consts.train_data)
    parser.add_argument("--devset", type=str, default=consts.dev_data)
    parser.add_argument("--testset", type=str, default=consts.test_data)

    hp = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Net(
        device=device,
        trigger_size=len(all_triggers),
        word_size=[len(word2id), consts.WORD_DIM],
        word_emb=wordemb,
        entity_size=[len(all_entities), consts.ENTITY_DIM],
        postags_size=[len(all_postags), consts.POSTAG_DIM],
        position_size=[2*consts.MAXLEN, consts.POSITION_DIM]
    )
    if device == 'cuda':
        model = model.cuda()

    train_dataset = ACE2005Dataset(hp.trainset)
    dev_dataset = ACE2005Dataset(hp.devset)
    test_dataset = ACE2005Dataset(hp.testset)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)
    dev_iter = data.DataLoader(dataset=dev_dataset,
                               batch_size=hp.batch_size,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=pad)
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    criterion = nn.CrossEntropyLoss()

    if not os.path.exists(hp.logdir):
        os.makedirs(hp.logdir)

    for epoch in range(1, hp.n_epochs + 1):
        print("=========train at epoch={}=========".format(epoch))
        train(model, train_iter, optimizer, criterion)

        fname = os.path.join(hp.logdir, str(epoch))
        print("=========eval dev at epoch={}=========".format(epoch))
        metric_dev = eval(model, dev_iter, fname + '_dev', write=False)

    
        print("=========eval test at epoch={}=========".format(epoch))
        metric_test = eval(model, test_iter, fname + '_test',write=False)

        torch.save(model, "best_model.pt")