import os
import sys
import pickle
import torch
import random

from pytorch_transformers import BertTokenizer, BertForQuestionAnswering
from pytorch_transformers import AdamW, WarmupLinearSchedule

from dataset import Dataset
from util import transfer_data_format, _build_event_ontology, build_examples, transfer_to_query_bert_format
from model_evaluate import model_evaluation

def load_model(model_path, device):
    model = BertForQuestionAnswering.from_pretrained(model_path).to(device)
    tokenizer_dir = '../data/my-bert-large-cased-squad/vocab.txt'
    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir, do_lower_case=False)
    return tokenizer, model


def train_portion(ace_ontology, data_ace, ratio):
    # parameters
    n_epoch = 10
    batch_size = 12

    learning_rate = 5e-5
    adam_epsilon = 1e-8
    warmup_steps = 0
    max_grad_norm = 1.0

    # load model
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = 'cuda'
    tokenizer, model = load_model('../data/my-bert-large-cased-squad/', device)
    # tokenizer, model = load_model('/home/jliu/data/BertModel/bert-large-cased', device)

    cut_idx = int(len(data_ace['train']) * ratio)
    print('Training examples', cut_idx)

    max_seq_len = 120
    training_set = build_examples(ace_ontology, data_ace['train'][:cut_idx], training=True)
    training_set = transfer_to_query_bert_format(training_set, tokenizer, max_seq_len, training=True)
    train_dataset = Dataset(batch_size, max_seq_len, training_set)

    # developping set
    dev_set = build_examples(ace_ontology, data_ace['dev'], training=False)
    dev_set = transfer_to_query_bert_format(dev_set, tokenizer, max_seq_len, training=False)
    dev_dataset = Dataset(batch_size, max_seq_len, dev_set)

    t_total = int(n_epoch * len(training_set) / batch_size)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
    
    torch.cuda.empty_cache()

    global_step = 0
    for _ in range(n_epoch):
        for batch in train_dataset.get_tqdm(device, shuffle=True):
            global_step += 1
            model.train()
            input_ids, input_mask, segment_ids, start_positions, end_positions, token_to_orig_map, example = batch

            inputs = {'input_ids': input_ids,
                    'attention_mask':  input_mask,
                    'token_type_ids':  segment_ids,
                    'start_positions': start_positions,
                    'end_positions':   end_positions}
            outputs = model(**inputs)
            loss = outputs[0]
            loss = loss.mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step() 
            model.zero_grad()
        
        model.eval()
        with torch.no_grad():
            model_evaluation(model, dev_dataset, device)
            print()
    #model.destroy() 
    del model
    torch.cuda.empty_cache()

if __name__ == '__main__':
    
    # # dump data
    data_framenet = pickle.load(open('../data/data_framenet.pickle', 'rb'))
    data_ace = pickle.load(open('../data/data_ace.pickle', 'rb'))
    for f in ['train', 'test', 'val']: 
        data_ace[f] = transfer_data_format(data_ace[f])
    
    ace_ontology = _build_event_ontology(data_ace['train'] + data_ace['test'] + data_ace['val'])
    frame_ontology = _build_event_ontology(data_framenet)
    
    random.shuffle(data_ace['train'])
    temp = [ace_ontology, data_ace]
    pickle.dump(temp, open('../data/data_progress.pickle', 'wb'))
    

    ace_ontology, data_ace = pickle.load(open('../data/data_progress.pickle', 'rb'))
    ratio = float(sys.argv[1])
    print('Training ratio', ratio)
    train_portion(ace_ontology, data_ace, ratio)
