import math
import numpy as np

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

class Model(object):

    def __init__(self, keys):
        self.keys = keys;
        self.model = ''
        
   
    
    def train_by_counting(self, data): #Version number indexes instead
        modelparams = {} ##all parameters: hiddens, obs, pi, trans, emis
        for i in  range( len(self.keys) ):
            modelparams[i] = {}
        
        for name in data:
            
            ##hidden
            hiddencount = 0
            for c in data[name]['Z']: #not effecient i know
                
                if c not in modelparams[0]:
                    modelparams[0][c] = hiddencount #hidden
                    hiddencount+=1
            
                
            ##obs
            obscount = 0
            for c in data[name]['X']:
                
                if c not in modelparams[1]:
                    modelparams[1][c] = obscount
                    obscount+=1
            
            
        ##make last dicts
        X = len(modelparams[1])
        Z = len(modelparams[0])
        modelparams[2] = [0] * Z
        modelparams[3] = np.zeros(shape=(Z,Z))
        modelparams[4] = np.zeros(shape=(Z,X)) 
            
        for name in data:
            #count pi
            ### Fair to assume that pi[1] is 0 probably? As sequence wont begin in membrane
            modelparams[2][modelparams[0][data[name]['Z'][0]]] += 1
            
            
        
            
        for name in data:
            #count transis
            for z in range(len(data[name]['X'])-1):
                f,t = modelparams[0][data[name]['Z'][z]], modelparams[0][data[name]['Z'][z+1]]
                modelparams[3][f][t] += 1
                
                #count emis
            for z in range(len(data[name]['X'])):
                h,o =  modelparams[0][data[name]['Z'][z]], modelparams[1][data[name]['X'][z]]  ###Would be cool to do this stuff in a more python way
                modelparams[4][h][o] += 1
    
      
        a = np.array(modelparams[2])
        modelparams[2]= a/float(len(data))
        
        for v in range(Z):
            modelparams[3][v]/=np.sum(modelparams[3][v])
        
        for v in range (Z):
            modelparams[4][v] /= np.sum(modelparams[4][v])
        
        self.model = modelparams
        
        
        
        
        
    
    def hidden_states(self):
        return self.model['hidden'].keys()

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



   

    def __str__(self):
        """Prints the data"""
        # print the hidden states
        string = ''
        for i in range(len(self.keys)):
            string += str(self.keys[i]) + '\n'
            string += str(self.model[i]) + '\n'

        return string
