import math

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
        
    def load(self, filename):
        """Loads the data and returns a dictionary with some kind of helpful structure"""
        data = {}
        # just fetch the lines which are not empty
        with open(filename, 'r') as f:
            EOF = False
            while not EOF:
                l = f.readline().strip('\n')
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
                    data[self.keys[2]] = map(lambda x:zerolog(x), split_line(f.readline().strip('\n'), t=float) )
                    
                elif l == self.keys[3]: #'transitions'
                    # multiple lines
                    #format [[0.2,..], [...]]
                    data[self.keys[3]] =   [ split_line(f.readline().strip('\n'), t=float) for x in range(len(data[self.keys[0]].keys())) ]
                    for y in range(len(data[self.keys[0]].keys())):
                        data[self.keys[3]][y] = map(lambda x:zerolog(x),data[self.keys[3]][y])
                    
                elif l == self.keys[4]: #'emissions'
                    # multiple lines
                    #format [[0.2,..], [...]]
                    data[self.keys[4]] = [ split_line(f.readline().strip('\n'), t=float) for x in range(len(data[self.keys[0]].keys())) ]
                    for y in range(len(data[self.keys[0]].keys())):
                        data[self.keys[4]][y] = map(lambda x:zerolog(x),data[self.keys[4]][y])
                        
                    # this is supposed to be the last line of our data
                    EOF = True
        self.model = data


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



    def viterbi(x):
        #viterbi recursion. yay
        w = [0] * len(data['pi'])
        print w
        
        

    def __str__(self):
        """Prints the data"""
        # print the hidden states
        string = ''
        for key in self.keys:
            string += str(key) + '\n'
            string += str(self.model[key]) + '\n'

        return string
