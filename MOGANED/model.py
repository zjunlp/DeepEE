import torch
import torch.nn as nn
from utils import EmbeddingLayer, MultiLabelEmbeddingLayer, get_positions
from GCN import MOGCN

class Net(nn.Module):
    def __init__(self, trigger_size=None, word_size=None, word_emb=None, entity_size=None, postags_size=None, position_size=None, device=torch.device("cpu")):
        super().__init__()

        lstm_dim = 250
        gcn_dim = 150
        self.device = device
        self.word_embed = EmbeddingLayer(embedding_size=word_size, embedding_matrix=word_emb,
                                          device=device)
        self.entity_embed = MultiLabelEmbeddingLayer(num_embeddings=entity_size[0], embedding_dim=entity_size[1], device=device)

        self.postag_embed = EmbeddingLayer(embedding_size=postags_size, device=device)
        self.position_embed = EmbeddingLayer(embedding_size=position_size, device=device)

        self.rnn = nn.LSTM(lstm_dim, lstm_dim//2, num_layers=1, batch_first=True, bidirectional=True)

        self.mogcn = MOGCN(in_dim = lstm_dim, hidden_dim=gcn_dim, K=3, dropout=0.3, device=device, alpha=0.2)

        self.fc_trigger = nn.Sequential(
            nn.Linear(gcn_dim, trigger_size)
        )

    def predict_triggers(self, tokens_2d, entities_3d, postags_2d, seqlen_1d, adjm):
        tokens = torch.LongTensor(tokens_2d).to(self.device)
        postags_2d = torch.LongTensor(postags_2d).to(self.device)
        word_emb = self.word_embed(tokens)
        pos_emb = self.postag_embed(postags_2d)
        ent_emb = self.entity_embed(entities_3d)
        BATCH_SIZE, SEQ_LEN = tokens.shape[:]
        mask = torch.zeros(BATCH_SIZE, SEQ_LEN)
        for i in range(BATCH_SIZE):
            s_len = seqlen_1d[i]
            mask[i, 0:s_len] = 1
        mask = torch.FloatTensor(mask).to(self.device)

        adj = torch.stack([torch.sparse.FloatTensor(torch.LongTensor(adjmm[0]),
                                                     torch.FloatTensor(adjmm[1]),
                                                     torch.Size([3, SEQ_LEN, SEQ_LEN])).to_dense() for
                            adjmm in adjm])
        adj = adj.to(self.device)
        x_emb = torch.cat([word_emb, pos_emb, ent_emb], 2)
        positional_sequences = get_positions(BATCH_SIZE, SEQ_LEN)
        xx = []
        for i in range(SEQ_LEN):
            # encoding
            posi_emb = self.position_embed(positional_sequences[i].to(self.device))
            lstm_input = torch.cat([x_emb, posi_emb],dim=2)
            x, _ = self.rnn(lstm_input)  # (batch_size, seq_len, d')
            # gcns
            gcn_o = self.mogcn(adj, x) #[batch_size, SEQ_LEN, d]

            xx.append(gcn_o[:,i,:])  # (batch_size, d')
        # output linear
        xx = torch.stack(xx, dim=1)  # (batch_size, seq_len, d')
        xx_m = torch.mul(xx, mask.unsqueeze(2))
        trigger_logits = self.fc_trigger(xx_m)
        trigger_hat_2d = trigger_logits.argmax(-1)

        return trigger_logits, trigger_hat_2d