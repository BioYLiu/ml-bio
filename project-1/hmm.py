
# coding: utf-8

# # Hidden Markov Models

###
#
# Computing the hidden markov models 
# Martin Bjerrum Henriksen
# Juan Francisco Marín Vega
# 
###

import math

MODEL_NAME = 'test.hmm'
SEQUENCES_NAME = 'sequences.txt'
KEYS = ['hidden', 'observables', 'pi', 'transitions', 'emissions']


def split_line(l, t=str, c=' '):
    """Returns an array of values from a line"""
    return  [ t(x) for x in l.split(c) ]


def dict_with_indexes(l):
    """Returns a dictionary with key:index from an array"""
    return { l[i]: i for i in range(len(l)) }


def load_model(filename):
    """Loads the data and returns a dictionary with some kind of helpful structure"""
    data = {}
    # just fetch the lines which are not empty
    with open(filename, 'r') as f:
        EOF = False
        while not EOF:
            l = f.readline().strip('\n')
            if l == KEYS[0]: #'hidden'
                # one line
                # format {'i':0, ...}
                data[KEYS[0]] = dict_with_indexes( split_line(f.readline().strip('\n')) ) 
            elif l == KEYS[1]: #'observables'
                # one line
                # format {'a':0,...}
                data[KEYS[1]] = dict_with_indexes( split_line(f.readline().strip('\n')) ) 
            elif l == KEYS[2]: #'pi'
                # one line
                # format [0.3, ...]
                data[KEYS[2]] = split_line(f.readline().strip('\n'), t=float)
            elif l == KEYS[3]: #'transitions'
                # multiple lines
                #format [[0.2,..], [...]]
                data[KEYS[3]] = [ split_line(f.readline().strip('\n'), t=float) for x in range(len(data[KEYS[0]].keys())) ] 
            elif l == KEYS[4]: #'emissions'
                # multiple lines
                #format [[0.2,..], [...]]
                data[KEYS[4]] = [ split_line(f.readline().strip('\n'), t=float) for x in range(len(data[KEYS[0]].keys())) ]    
                # this is supposed to be the last line of our data
                EOF = True      
    return data


def load_sequences(filename):
    data = {}
    # just fetch the lines which are not empty
    with open(filename, 'r') as f:
        lines = f.readlines()
        x = 0
        while x < len(lines):
            name = lines[x].strip('\n').lstrip('>')
            x += 1
            emissions = lines[x].strip('\n').lstrip(' ')
            x += 1
            hiddens = lines[x].strip('\n').lstrip('# ')
            x += 2
            data[name] = dict(X=emissions, Z=hiddens, name=name)
    return data




def transition(a, b):
    """a and b are indexes of hidden states"""
    return model['transitions'][a][b]


def emission(a, b):
    """a is an index of a hidden state, b is an index of an emission"""
    return model['emissions'][a][b]


def compute_hmm(model, sequence):

    input_states = sequence['Z']
    input_emissions = sequence['X']
    states_indexes = model['hidden']
    observables_indexes = model['observables']
    # the first state
    i = 0
    PI = math.log( model['pi'][ states_indexes[input_states[i]] ] )
    S = PI
    # first hidden and emission nodes
    S += math.log( emission(states_indexes[input_states[i]], observables_indexes[input_emissions[i]]) )
    for i in range(1, len(input_states)):
        S += math.log( transition(states_indexes[input_states[i - 1]], states_indexes[input_states[i]]) )
        S += math.log( emission(states_indexes[input_states[i]], observables_indexes[input_emissions[i]]) )
        
    return (sequence['name'], S)
        

def show_model(data):
    """Prints the model"""
    # print the hidden states
    for key in KEYS:
        print key
        print data[key]


def show_results(data):
    """Prints the results"""
    # print the hidden states
    for key in data:
        print key
        print "log P(x,z) =  %s" % data[key]


    
if __name__ == "__main__":
    
    # loading the model
    model = load_model(MODEL_NAME)
    # loading the sequence
    sequences = load_sequences(SEQUENCES_NAME)

    # computing the model
    results = {}
    for key in sequences.keys():
        results[key] =  compute_hmm(model, sequences[key])[1]



    print "Model:\n"
    show_model(model)
    print "Results"
    show_results(results)


