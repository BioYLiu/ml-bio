import math
import numpy as np
from viterbi import Viterbi

def split_line(l, t=str, c=' '):
    """Returns an array of values from a line"""
    return  [ t(x) for x in l.split(c) ]


def dict_with_indexes(l):
    """Returns a dictionary with key:index from an array"""
    return { l[i]: i for i in range(len(l)) }

def zerolog(f):
    if (f==0.0):
        return -float('inf')
    return math.log(f)

def infexp(f):
    if f == -float('inf'):
        return 0.0
    return math.exp(f)
    
    


class Model(object):
    
    def __init__(self, keys, base_model=None, labels=None):
        self.keys = keys
        self.model = base_model or ''
        self.labels = labels
   
    def delog(self):
        
        #pi
        self.model['pi'] = [ infexp(x) for x in self.model['pi'] ]
        # transitions
        for i in range(len(self.model['transitions'])):
            self.model['transitions'][i] = [ infexp(x) for x in self.model['transitions'][i] ]
            
        # emissions
        for i in range(len(self.model['emissions'])):
            self.model['emissions'][i] = [ infexp(x) for x in self.model['emissions'][i] ]
 
            
    def log_space(self):
        
        #pi
        self.model['pi'] = [ zerolog(x) for x in self.model['pi'] ]
        # transitions
        for i in range(len(self.model['transitions'])):
            self.model['transitions'][i] = [ zerolog(x) for x in self.model['transitions'][i] ]
            
        # emissions
        for i in range(len(self.model['emissions'])):
            self.model['emissions'][i] = [ zerolog(x) for x in self.model['emissions'][i] ]
            
        
    

    def train_by_counting(self, data):
        """
        Computes training by counting for 3 states i M o
        :param data: a dictionary of sequences
        :return:
        """
        

        ##make last dicts
        Z = len(self.model['hidden'])
        X = len(self.model['observables'])
        
        # let's count!
        for name in data:
            hiddens = np.array(list(data[name]['Z']))
            observables = np.array(list(data[name]['X']))
            #count pi
            ### Fair to assume that pi[1] is 0 probably? As sequence wont begin in membrane
            self.model['pi'][self.index_hidden_state(hiddens[0])] += 1


            #count transmissions

            for z in range(len(hiddens) - 1):
                # from hidden f to hidden t
                f, t = self.index_hidden_state(hiddens[z]), self.index_hidden_state(hiddens[z + 1])
                self.model['transitions'][f][t] += 1

            #count emissions
            for z in range(len(observables)):
                # from hidden h to observable o
                h, o =  self.index_hidden_state(hiddens[z]), self.index_observable(observables[z])
                self.model['emissions'][h][o] += 1

        # normalizing PI
        self.model['pi'] = np.array(self.model['pi']) / float(len(data))
        
        ###forgot about logs!
        #self.model['pi'] =  map(lambda x:zerolog(x),self.model['pi'])

        # normalize and logify transitions row by row
        for v in range(Z):
            self.model['transitions'][v] /= np.sum(self.model['transitions'][v])
            #self.model['transitions'][v] = map(lambda x:zerolog(x),self.model['transitions'][v])

        #normalize and logify emissions row by row
        for v in range (Z):
            self.model['emissions'][v] /= np.sum(self.model['emissions'][v])
            #self.model['emissions'][v] =  map(lambda x:zerolog(x),self.model['emissions'][v])


        # if the model has labels, we need to translate back the sequence of Z's
        if self.labels is not None:
            for seq in data.values():
                seq['Z'] = self.translate_hidden_states(seq['Z'])

        ## now the function returns a model!!
        #print self.model
        return Model(self.keys, self.model, self.labels)

    def load(self, filename):
        """Loads the data and returns a dictionary with some kind of helpful structure"""
        data = {}
        # just fetch the lines which are not empty
        with open(filename, 'r') as f:
            EOF = False
            while not EOF:
                l = f.readline().strip('\n')
                print l
                if l == self.keys[0]: #'hidden'
                    # one line
                    # format {'i':0, ...}
                    data[self.keys[0]] = dict_with_indexes( split_line(f.readline().strip('\n')) )
                elif l == self.keys[1]: #'observables'
                    # one line
                    # format {'a':0,...}
                    data[self.keys[1]] = dict_with_indexes( split_line(f.readline().strip('\n')) )
                elif l == self.keys[2]: #'pi'
                    # one line
                    # format [0.3, ...]
                    #a =  split_line(f.readline().strip('\n'), t=float) 
                    #print type(a[0])
                    data[self.keys[2]] =  split_line(f.readline().strip('\n'), t=float)
                    
                elif l == self.keys[3]: #'transitions'
                    # multiple lines
                    #format [[0.2,..], [...]]
                    data[self.keys[3]] =   [ split_line(f.readline().strip('\n'), t=float) for x in range(len(data[self.keys[0]].keys())) ]
                    #for y in range(len(data[self.keys[0]].keys())):
                    #    data[self.keys[3]][y] = data[self.keys[3]][y]
                    
                
                    EOF = True
            
                
        self.model = data
        #self.model[self.keys[4]] = [ [1.0] * len(self.model['observables'])  ] * len(self.hidden_states())
        self.model[self.keys[4]] = np.random.random_sample((len(self.hidden_states()), len(self.model['observables'])))
        
        # normalize pi
        self.model['pi'] = np.array(self.model['pi'])
        self.model['pi'] /= np.sum(self.model['pi'])
        
        # normalize transitions
        for i in range(len(self.model['transitions'])):
            self.model['transitions'][i] = np.array(self.model['transitions'][i])
            self.model['transitions'][i] /= np.sum(self.model['transitions'][i])
            
        
        # normalize emissions
        for i in range(len(self.model['emissions'])):
            self.model['emissions'][i] = np.array(self.model['emissions'][i])
            self.model['emissions'][i] /= np.sum(self.model['emissions'][i])
    
    
    def train(self, sequences, iterations=3):
        vit = Viterbi()
        for x in range(iterations):
            self.log_space()
            for name, seq in sequences.items():
                seq['Z'] = vit.decode(self, seq['X'])
                print seq['Z']
            #we return from log space
            self.delog()
            self.train_by_counting( sequences )
            print Model(self.keys, self.model, self.labels) 
        
        return Model(self.keys, self.model, self.labels) 
        
        
        
    def hidden_states(self):
        # here we had self.model['hidden'].keys()
        return sorted(self.model['hidden'], key=self.model['hidden'].get)

    def emission(self, a, b):
        """a is an index of a hidden state, b is an index of an emission"""
        return self.model['emissions'][a][b]

    def transition(self, a, b):
        """a and b are indexes of hidden states"""
        return self.model['transitions'][a][b]

    def index_observable(self, a):
        """returns the index in the model of a, when a is an observable """
        return self.model['observables'][a]

    def index_hidden_state(self, a):
        """returns the index in the model of a, when a is a hidden state"""
        return self.model['hidden'][a]
        
    def pi(self, a):
        """a is a index of pi"""
        return self.model['pi'][a]

    def get_labels(self):
        return self.labels

    def translate_hidden_states(self, sequence):
        return "".join([self.labels[x] for x in sequence])



   

    def __str__(self):
        """Prints the data"""
        string = ''
        if type(self.model.keys()[0]) != str:
            for i in range(len(self.keys)):
                string += str(self.keys[i]) + '\n'
                string += str(self.model[i]) + '\n'
        else:
            for key in self.keys:
                string += key + '\n'
                string += str(self.model[key]) + '\n'



        return string
