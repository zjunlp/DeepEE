import os
import pickle
import torch
import random

from pytorch_transformers import BertTokenizer, BertForQuestionAnswering
from pytorch_transformers import AdamW, WarmupLinearSchedule

from dataset import Dataset
from util import transfer_data_format, _build_event_ontology, build_examples, transfer_to_query_bert_format, evaluate

def load_model(model_path, device):
    model = BertForQuestionAnswering.from_pretrained(model_path).to(device)
    tokenizer_dir = '/home/jliu/data/BertModel/bert-large-cased-squad/bert-large-cased-whole-word-masking-finetuned-squad-vocab.txt'
    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir, do_lower_case=False)
    return tokenizer, model


def model_evaluation(model, dataset, device):
    examples, predicted_s, predicted_e, token_to_orig_maps = list(), list(), list(), list()
    for batch in dataset.get_tqdm(device, shuffle=False):
        input_ids, input_mask, segment_ids, _, _, token_to_orig_map, example = batch

        inputs = {'input_ids': input_ids,
                'attention_mask':  input_mask,
                'token_type_ids':  segment_ids}
        outputs = model(**inputs)
        
        examples.extend(example)
        predicted_s.extend(outputs[0].cpu().numpy())
        predicted_e.extend(outputs[1].cpu().numpy())
        token_to_orig_maps.extend(token_to_orig_map)

    evaluate(examples, predicted_s, predicted_e, token_to_orig_maps, entity_refine=False)
    evaluate(examples, predicted_s, predicted_e, token_to_orig_maps)


def modify_framenet(data):
    results = list()
    for elem in data:
        args = elem[-1]
        for arg in args:
            s, e = arg[1], arg[2]
            if e - s > 3:
                break
        else:
            elem.insert(3, {})
            results.append(elem)
    return results


if __name__ == '__main__':

    batch_size = 12

    # load model
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda'
    tokenizer, model = load_model('../large_cased_finetuned', device)

    # load ACE data
    data_ace = pickle.load(open('../data/data_ace.pickle', 'rb'))
    for f in ['train', 'test', 'val']: 
        data_ace[f] = transfer_data_format(data_ace[f])
    ace_ontology = _build_event_ontology(data_ace['train'] + data_ace['test'] + data_ace['val'])
    print(ace_ontology)
    max_seq_len = 120
    testing_set = build_examples(ace_ontology, data_ace['test'], training=False)
    testing_set = transfer_to_query_bert_format(testing_set, tokenizer, max_seq_len, training=False)
    test_dataset = Dataset(batch_size, max_seq_len, testing_set)
    
    
    data_framenet = pickle.load(open('../data/data_framenet.pickle', 'rb'))
    frame_ontology = _build_event_ontology(data_framenet)
    data_framenet = modify_framenet(data_framenet)

    max_seq_len = 120
    testing_set = build_examples(frame_ontology, data_framenet[:500], training=False)
    testing_set = transfer_to_query_bert_format(testing_set, tokenizer, max_seq_len, training=False)
    test_dataset = Dataset(batch_size, max_seq_len, testing_set)

    model.eval()
    with torch.no_grad():
        model_evaluation(model, test_dataset, device)
