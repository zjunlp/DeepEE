import os
import torch
from data_load import idx2trigger
from utils import calc_metric, find_triggers


def eval(model, iterator, fname, write):
    model.eval()

    words_all, triggers_all, triggers_hat_all = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            tokens_2d, triggers_2d, entities_3d, postags_2d, adj, seqlen_1d, words, triggers = batch

            trigger_logits, trigger_hat_2d = model.predict_triggers(tokens_2d=tokens_2d, entities_3d=entities_3d,
                                                                    postags_2d=postags_2d, seqlen_1d=seqlen_1d,
                                                                    adjm=adj)
            words_all.extend(words)
            triggers_all.extend(triggers)
            triggers_hat_all.extend(trigger_hat_2d.cpu().numpy().tolist())

    triggers_true, triggers_pred = [], []
    with open('temp', 'w') as fout:
        for i, (words, triggers, triggers_hat) in enumerate(zip(words_all, triggers_all, triggers_hat_all)):
            triggers_hat = triggers_hat[:len(words)]
            triggers_hat = [idx2trigger[hat] for hat in triggers_hat]

            # [(ith sentence, t_start, t_end, t_type_str)]
            triggers_true.extend([(i, *item) for item in find_triggers(triggers)])
            triggers_pred.extend([(i, *item) for item in find_triggers(triggers_hat)])

            for w, t, t_h in zip(words, triggers, triggers_hat):
                fout.write('{}\t{}\t{}\n'.format(w, t, t_h))
            fout.write("\n")


    print('[trigger classification]')
    trigger_p, trigger_r, trigger_f1 = calc_metric(triggers_true, triggers_pred)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(trigger_p, trigger_r, trigger_f1))
    print('[trigger identification]')
    triggers_true = [(item[0], item[1], item[2]) for item in triggers_true]
    triggers_pred = [(item[0], item[1], item[2]) for item in triggers_pred]
    trigger_p_, trigger_r_, trigger_f1_ = calc_metric(triggers_true, triggers_pred)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(trigger_p_, trigger_r_, trigger_f1_))

    metric = '[trigger classification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(trigger_p, trigger_r, trigger_f1)
    metric += '[trigger identification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(trigger_p_, trigger_r_, trigger_f1_)
    final = fname
    if write:
        with open(final, 'w') as fout:
            result = open("temp", "r").read()
            fout.write("{}\n".format(result))
            fout.write(metric)
        os.remove("temp")
    return metric
