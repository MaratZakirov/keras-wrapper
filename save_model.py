#  -*- coding: utf-8 -*-
import numpy
import json
from keras import backend as K
from keras.models import load_model

model = load_model('query.bin')
model.compile(optimizer='sgd', loss='mse')
open('model-structure.json', 'w').write(json.dumps(json.loads(model.to_json()), indent=4, sort_keys=True))

def ArrToStr(arr):
    s = []
    if len(arr.shape) == 2:
        for i in xrange(arr.shape[0]):
            s_2 = []
            for j in xrange(arr.shape[1]):
                s_2.append(str(arr[i][j]))
            s_2 = ' '.join(s_2)
            s.append(s_2)
        s = '\n'.join(s)
    else:
        for i in xrange(arr.shape[0]):
            s.append(str(arr[i]))
        s = ' '.join(s)
    return s

def SaveData(data, filepath):
    data = numpy.reshape(data, (data.shape[1], data.shape[2]))
    fout = open(filepath, 'w')
    for i in xrange(data.shape[0]):
        if numpy.sum(data[i]) == 0:
            continue
        for j in xrange(data.shape[1]):
            fout.write(str(data[i][j]) + ' ')
        fout.write('\n')

def SaveWeights(model, filepath):
    root = {}

    if hasattr(model, 'flattened_layers'):
        # Support for legacy Sequential/Merge behavior.
        flattened_layers = model.flattened_layers
    else:
        flattened_layers = model.layers

    for layer in flattened_layers:
        #print layer.name
        root[layer.name] = {}
        symbolic_weights = layer.weights
        weight_values = K.batch_get_value(symbolic_weights)
        weight_names = []
        for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
            if hasattr(w, 'name') and w.name: name = str(w.name)
            else:                             name = 'param_' + str(i)
            weight_names.append(name.encode('utf8'))
        for name, val in zip(weight_names, weight_values):
            root[layer.name][name] = ArrToStr(val)
            #print name
            #print val

    #print json.dumps(root, indent=4, sort_keys=True)
    open(filepath, 'w').write(json.dumps(root, indent=4, sort_keys=True))

def loadtable(file):
    table = {}
    for i, line in enumerate(open(file).readlines()):
        word = line.split()[0]
        word = word.decode('utf-8')
        table[word] = i
    return table

def setVec(vec, word, table):
    inc_j = False
    word = word.decode('utf-8')
    word = '#' + word + '#'
    for i in xrange(len(word) - 1):
        e = word[i] + word[i + 1]
        if e in table:
            vec[table[e]] += 1.0
            inc_j = True
    return inc_j

def loadStr(str, table, S):
    for i, word in enumerate(str.split()):
        setVec(S[i], word, table)

wd = '.'
TABLE = wd + 'bigrams.txt'
table = loadtable(TABLE)
featnum = len(table)

s_t_r = 'some cool query'
s_t_r = s_t_r.encode('utf-8')
data = numpy.zeros(shape=(1, 10, featnum))
loadStr(s_t_r, table, data[0])

print 'Perform: ', model.predict(x=data)

SaveWeights(model, 'model-weights.json')
SaveData(data, 'data.txt')
