def generate_context(flag):
    result = []
    data_ace = pickle.load(open('../data/data_ace.pickle', 'rb'))
    for f in ['train', 'test', 'val']:
        data_ace[f] = transfer_data_format(data_ace[f])
    
    for elem in data_ace[flag]:
        print(elem[2])
        words = elem[2]
        trigger_span = elem[4]
        trigger_start = max(0, int(trigger_span[0]) - 5)
        trigger_end = int(trigger_span[1]) + 5
        result.append(' '.join(words[trigger_start:trigger_end]))
    
    if flag == 'test':
        return result
    
    data_framenet = pickle.load(open('../data/data_framenet.pickle', 'rb'))[:10000]

    for elem in data_framenet:
        words = elem[2]
        trigger_span = elem[3]
        trigger_start = max(0, int(trigger_span[0]) - 5)
        trigger_end = int(trigger_span[1]) + 5
        result.append(' '.join(words[trigger_start:trigger_end]))
    return result


def generate_question():
    # Squad Question
    result = []
    squad_data = json.loads(open('/home/jliu/data/squad/train-v2.0.json').read())
    squad_data = squad_data['data']
    for data in squad_data:
        for elem in data['paragraphs']:
            qas, context = elem['qas'], elem['context']
            for q in qas:
                question = q['question']
                question = question.split()
                try:
                    if (question[0] == 'What' or question[0] == 'Who') and (question[1] == 'is' or question[1] == 'are') and question[2] == 'the':
                        temp = ' '.join(question[4:])
                        if temp: result.append(temp)
                except:
                    pass
                if question[0] == 'Where' or question[0] == 'When':
                    temp = ' '.join(question[1:])
                    if temp: result.append(temp)
    
    print('Squad', len(result))

    # wiki.answers.com
    from os import listdir
    from os.path import isfile, join
    mypath = '../data/QA'
    onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    for f in onlyfiles:
        try:
            data = json.loads(open(f).read())
            for d in data:
                question = d['Question']
                question = question.split()
                if len(question) > 20: continue
                try:
                    if (question[0] == 'What' or question[0] == 'Who') and (question[1] == 'is' or question[1] == 'are') and question[2] == 'the':
                        temp = ' '.join(question[4:])
                        if temp: result.append(temp)
                except:
                    pass
                if question[0] == 'Where' or question[0] == 'When':
                    temp = ' '.join(question[1:])
                    if temp: result.append(temp)
        except:
            pass
    print('All', len(result))
    return result


if __name__ == '__main__':
	fin = open('../data/src_context.txt', 'w')
    train_context = generate_context('train')
    print(train_context[0])
    print('\n'.join(train_context), file=fin)
    fin.close()
    

    fin = open('../data/tgt_question.txt', 'w')
    questions = generate_question()
    print('\n'.join(questions), file=fin)
    fin.close()
    