import random
import pickle
import torch
import os
import numpy as np
import json

from pytorch_transformers import BertTokenizer, BertForQuestionAnswering

from dataset import Dataset

class eval_object(object):
    def __init__(self):
        self.predict = 1e-200
        self.gold = 1e-200
        self.correct = 1e-200


def _get_position(bio_list, tag):
    s, e = -1, -1
    t = 'null'
    for idx, label in enumerate(bio_list):
        if label[0] == 'B' and label.split(':')[1] == tag:
            s = idx
            e = idx
            t = label.split(':')[0][2:]
        if label[0] == 'I' and label.split(':')[1] == tag:
            e = idx
    e = e + 1
    return s, e, t

def _build_event_ontology(datasets):
    event_ontology = {}
    for data in datasets:
        event_type = data[0]
        args = data[-1]
        for arg in args:
            event_ontology.setdefault(event_type, set())
            event_ontology[event_type].add(arg[0])
    return event_ontology


def transfer_data_format(ace_dataset):
    results = []
    for data in ace_dataset:
        doc_id, text, entity_head_list, ner_total_list, event_list, events, pos, related_entities = data
        for event_id in events:
            event_arguments = events[event_id]
            trigger_s, trigger_t, event_type = _get_position(event_list, event_id)
            #print(event_arguments)
            event_arguments_list = list()
            for argument in event_arguments:
                argument_role, a_id = argument
                # argument_s, argument_t, _ = _get_position(entity_head_list, a_id)
                if argument_role.startswith('Time'):
                    argument_role = 'Time'
                if a_id not in related_entities:
                    print(a_id, doc_id)
                    continue
                s = related_entities[a_id][4]
                t = related_entities[a_id][5] + 1
                event_arguments_list.append([argument_role, s, t])
            event_pos = [trigger_s, trigger_t]
            results.append([event_type, event_type, text, related_entities, event_pos, event_arguments_list])
    return results


def build_bert_example(tokenizer, query, context, start_pos, end_pos, max_seq_length,
                    cls_token='[CLS]', sep_token='[SEP]', pad_token=0, 
                    sequence_a_segment_id=0, sequence_b_segment_id=1,
                    cls_token_segment_id=0, pad_token_segment_id=0):
    
    is_impossible = (start_pos == -1)
    
    query_tokens = tokenizer.tokenize(query)

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(context):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_start_position = -1
    tok_end_position = -1

    tok_start_position = orig_to_tok_index[start_pos]
    tok_end_position = orig_to_tok_index[end_pos] - 1

    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    tokens = []
    token_to_orig_map = {}
    segment_ids = []

    tokens.append(cls_token)
    segment_ids.append(cls_token_segment_id)
    cls_index = 0

    for token in query_tokens:
        tokens.append(token)
        segment_ids.append(sequence_a_segment_id)

    tokens.append(sep_token)
    segment_ids.append(sequence_a_segment_id)

    for i in range(len(all_doc_tokens)):
        token_to_orig_map[len(tokens)] = tok_to_orig_index[i]
        tokens.append(all_doc_tokens[i])
        segment_ids.append(sequence_b_segment_id)

    # SEP token
    tokens.append(sep_token)
    segment_ids.append(sequence_b_segment_id)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(pad_token)
        input_mask.append(0)
        segment_ids.append(pad_token_segment_id)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    
    if is_impossible:
        start_position = cls_index
        end_position = cls_index
    else:
        doc_offset = len(query_tokens) + 2
        start_position = tok_start_position + doc_offset
        end_position = tok_end_position + doc_offset
    
    return input_ids, input_mask, segment_ids, token_to_orig_map, start_position, end_position


def build_examples(event_ontology, event_data, training=True):
    results = []
    for data in event_data:
        event_type, _, text, related_entities, trigger_pos, argument_colection = data
        argument_set = event_ontology[event_type]
        for arg in argument_set:
            arg_pos = list()
            for elem in argument_colection:
                at, s, e = elem
                if at == arg:
                    arg_pos.append([s, e])
            if len(arg_pos) > 1:
                # print(at, arg_pos)
                pass
            arg_pos = [[-1, -1]] if len(arg_pos) == 0 else arg_pos
            if training:
                for elem in arg_pos:
                    results.append([arg, text, related_entities, trigger_pos, elem])
            else:
                results.append([arg, text, related_entities, trigger_pos, arg_pos])
    return results


# def compute_prf(predictions):
#     num_predicted = 1e-50
#     num_corrected = 1e-50
#     num_golden = 1e-50

#     for elem in predictions:
#         g1, g2, p1, p2 = elem
#         if g1 != -1:
#             num_golden += 1
#         if p1 != -1:
#             num_predicted += 1
#         if (g1 == p1 or g2 == p2) and (g1 != -1 and g2 != -1):
#             num_corrected += 1
    
#     precision = num_corrected / num_predicted
#     recall = num_corrected / num_golden
#     f1 = 2 * precision * recall / (precision + recall)
#     print(num_corrected, num_predicted, num_golden, precision, recall, f1)



def evaluate(examples, predicted_s, predicted_e, token_to_orig_maps, entity_refine=True):

    type_eval = {}

    num_predicted = 1e-200
    num_corrected = 1e-200
    num_golden = 1e-200

    num_corrected_partial = 1e-200

    related_entites = [example[2] for example in examples]

    golden_list = [example[-1] for example in examples]  
    for i in range(0, len(golden_list)):
        golden_list[i] = list(filter(lambda x: x[0] != -1, golden_list[i]))
        golden_list[i] = list(map(lambda x: [x[0], x[1]-1], golden_list[i])) ##### note -1 here ##### 

    
    golden_roles = [example[0] for example in examples]  

    logit_start, logit_end = np.asarray(predicted_s), np.asarray(predicted_e)
    start_sorted, end_sorted = np.argsort(-logit_start, axis=1), np.argsort(-logit_end, axis=1)

    for start, end, output_0, output_1, golden, token_to_orig_map, entity_map, g_role in zip(start_sorted, end_sorted, logit_start, logit_end, golden_list, token_to_orig_maps, related_entites, golden_roles):
        res = list()
        for s in start:
            if s == 0: break
            for e in end:
                if e == 0: break
                if e < s: continue
                if e - s > 4:
                    continue
                res.append([s, e, output_0[s] + output_1[e]])
        res = sorted(res, key=lambda x: x[-1], reverse=True)
        def _to_org(x): return token_to_orig_map[x] if x in token_to_orig_map else -1
        res = list(map(lambda x: [_to_org(x[0]), _to_org(x[1]), x[2]], res))
        res = list(filter(lambda x: x[0] != -1 and x[1] != -1 and x[2] > 0, res))
        
        # print(entity_map)

        type_eval.setdefault(g_role, eval_object())

        # predicted = res[0:1]
        predicted = list()
        for elem in res:
            if not entity_refine:   ## entity refine??
                predicted.append(elem)
            else:
                for key in entity_map:
                    _, _, _, _, s, t = entity_map[key]
                    if elem[0] == s:
                        predicted.append([s, t])
                        continue
                    if elem[1] == t:
                        predicted.append([s, t])
                        continue
                    if elem[0] > s and elem[0] < t:
                        predicted.append([s, t])
                        continue
                    if elem[1] > s and elem[1] < t:
                        predicted.append([s, t])
                        continue
        golden_set = set()
        for elem in golden: golden_set.add(tuple(elem))

        predicted_set = set()
        for elem in predicted: predicted_set.add(tuple([elem[0], elem[1]]))
                
        num_golden += len(golden_set)
        num_predicted += len(predicted_set)
        num_corrected += len(golden_set.intersection(predicted_set))

        type_eval[g_role].predict += len(golden_set)
        type_eval[g_role].gold += len(predicted_set)
        type_eval[g_role].correct += len(golden_set.intersection(predicted_set))

        for g in golden_set:
            for p in predicted_set:
                if g[0] == p[0] or g[1] == p[1]:
                    num_corrected_partial += 1
                    break
    
    precision = num_corrected / num_predicted
    recall = num_corrected / num_golden
    f1 = 2 * precision * recall / (precision + recall)
    print(num_corrected, num_predicted, num_golden, precision, recall, f1)

    
    #print()
    # # Partial
    # precision = num_corrected_partial / num_predicted
    # recall = num_corrected_partial / num_golden
    # f1 = 2 * precision * recall / (precision + recall)
    # print(num_corrected_partial, num_predicted, num_golden, precision, recall, f1)

    res = list()
    for g_role in type_eval:
        temp = type_eval[g_role]
        precision = temp.correct / temp.predict
        recall = temp.correct / temp.gold
        f1 = 2 * precision * recall / (precision + recall)
        res.append([g_role, precision, recall, f1])
    res = sorted(res, key=lambda x: x[-1], reverse=True)
    for elem in res:
        print(*elem)
    print()



def build_query(arg, trigger_word):
    if arg in ['Attacker', 'Person', 'Victim', 'Buyer',
        'Giver', 'Seller', 'Benefciary', 'Defendant',
        'Prosecutor', 'Adjudicator', 'Plaintiff']:
        return 'Who is the %s in the %s?' % (arg.lower(), trigger_word)
    if arg in ['Time']:
        return 'When the %s occurs?' % (trigger_word)
    if arg in ['Where']:
        return 'Where the %s occurs?' % (trigger_word)
    
    return 'What is the %s of %s?' % (arg.lower(), trigger_word)


def transfer_to_query_bert_format(dataset, tokenizer, max_seq_len, training=True):
    all_examples = list()
    for idx, example in enumerate(dataset):
        idx += 1
        if idx % 500 == 0:
            print(idx, '...')
        arg, text, related_entities, trigger_pos, arg_pos = example
        query = build_query(arg, ' '.join(text[trigger_pos[0]: trigger_pos[1]]))
        try:
            if training:
                input_ids, input_mask, segment_ids, token_to_orig_map, start_position, end_position = build_bert_example(
                    tokenizer, query, text, arg_pos[0], arg_pos[1], max_seq_len
                )
            else:
                input_ids, input_mask, segment_ids, token_to_orig_map, start_position, end_position = build_bert_example(
                    tokenizer, query, text, -1, -1, max_seq_len
                )
        except Exception as e:
            print(e)
            continue
        all_examples.append([input_ids, input_mask, segment_ids, token_to_orig_map, start_position, end_position, example])
    return all_examples




if __name__ == '__main__':


    data_ace = pickle.load(open('../data/data_ace.pickle', 'rb'))
    for f in ['train', 'test', 'val']:
        data_ace[f] = transfer_data_format(data_ace[f])
    ace_ontology = _build_event_ontology(data_ace['train'] + data_ace['test'] + data_ace['val'])
    
    training_set = build_examples(ace_ontology, data_ace['train'], training=True)
    testing_set = build_examples(ace_ontology, data_ace['test'], training=False)


    # # framenet examples
    data_framenet = pickle.load(open('../data/data_framenet.pickle', 'rb'))
    frame_ontology = _build_event_ontology(data_framenet)
    data_framenet_choice = random.choices(data_framenet, k=2500)
    training_set_framenet = build_examples(frame_ontology, data_framenet_choice, training=False)

   
