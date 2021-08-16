import nltk
import json
import pickle
from nltk.corpus import framenet as fn

def extrac_framenet():
    results = []
    frames = fn.frames()
    for f in frames:
        temp = {}
        f_name = f.name
        f_definition = f.definition
        f_lexunit = f.lexUnit
        f_fes = f.FE

        temp['name'] = f_name
        temp['definition'] = f_definition
        temp['lexunit'] = list(f_lexunit.keys())
        temp['fes'] = [[fe, f_fes[fe].coreType, f_fes[fe].definition] for fe in f_fes]

        results.append(temp)

    file_object = open('../data/frame.json', 'w')
    json.dump(results, file_object)


def extract_examples():
    results = []
    frames = fn.frames()
    for f in frames:
        for lu in f.lexUnit:
            examples = f.lexUnit[lu].exemplars
            for example in examples:
                temp = {}
                temp['name'] = f.name
                temp['lexunit'] = lu
                temp['text'] = example.text
                if 'Target' in example:
                    temp['target'] = example.Target
                else:
                    print(example.text)
                temp['fe'] = example.FE
                results.append(temp)
                if len(results) % 100 == 0:
                    print('Processing...', len(results))

    file_object = open('../data/frame_examples.json', 'w')
    json.dump(results, file_object)

if __name__ == '__main__':
    # extrac_framenet()
    # extract_examples()
    with open('../data/frame_examples.json') as json_file:
        data = json.load(json_file)
        print(data[0])

    data = pickle.load(open('../data/data_framenet.pickle', 'rb'))
    print(data[0])
    data = pickle.load(open('../data/data_ace.pickle', 'rb'))
    for d in data['train']:
        if d[-1]:
            print(d)