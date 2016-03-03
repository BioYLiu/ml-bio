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

    def load(self, filename): ##REMOVE THIS
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
        
        
    def __init__(self, keys, base_model=None, labels=None):
        self.keys = keys
        self.model = base_model or ''
        self.labels = labels
   
    # I keep it to test the results
    def train_by_counting_old(self, data):
        """
        Computes training by counting for 3 states i M o
        :param data: a dictionary of sequences
        :return:
        """
        #Version number indexes instead
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

    def train_by_counting(self, data):
        """
        Computes training by counting for 3 states i M o
        :param data: a dictionary of sequences
        :return:
        """
        self.model = {x:{} for x in self.keys} ##all parameters: hiddens, obs, pi, trans, emissions

        # modified the first part because for each sequence
        # it's possible that not all the states are visited
        # and the same for the observables
        for name in data:
            hiddens = list(data[name]['Z'])
            observables = list(data[name]['X'])

            # getting all the different hiddens of each sequence
            # gets the different hiddens for this sequence, and its sums
            sub_hiddens = np.unique(hiddens, False) # return the uniques
            sub_observables = np.unique(observables, False)

            # updating the indexes for each state
            # caution!, not all states are visited in each sequence
            for i in range(len(sub_hiddens)):
                hidd = sub_hiddens[i]
                # get returns the second parameter if the key doesn't exists
                # so we add a new key with its new index
                # same as->  if hidd not in self.model['hidden]: self.model['hidden][hidd] = len(self.model['hidden']])
                self.model['hidden'][hidd] = self.model['hidden'].get(hidd, len(self.model['hidden']))

            for i in range(len(sub_observables)):
                sub = sub_observables[i]
                # get returns the second parameter if the key doesn't exists
                # so we add a new key with its new index
                self.model['observables'][sub] = self.model['observables'].get(sub, len(self.model['observables']))

        ### At this step  we have all the different 4 states with its index in self.model['hidden'] and
        ### all the different observables with its index in self.model['observables']


        ##make last dicts
        Z = len(self.model['hidden'])
        X = len(self.model['observables'])
        self.model['pi'] = [0] * Z
        self.model['transitions'] = np.zeros(shape=(Z,Z))
        self.model['emissions'] = np.zeros(shape=(Z,X))

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
        self.model['pi'] =  map(lambda x:zerolog(x),self.model['pi'])

        # normalize and logify transitions row by row
        for v in range(Z):
            self.model['transitions'][v] /= np.sum(self.model['transitions'][v])
            self.model['transitions'][v] = map(lambda x:zerolog(x),self.model['transitions'][v])

        #normalize and logify emissions row by row
        for v in range (Z):
            self.model['emissions'][v] /= np.sum(self.model['emissions'][v])
            self.model['emissions'][v] =  map(lambda x:zerolog(x),self.model['emissions'][v])


        # if the model has labels, we need to translate back the sequence of Z's
        if self.labels is not None:
            for seq in data.values():
                seq['Z'] = self.translate_hidden_states(seq['Z'])

        ## now the function returns a model!!
        return Model(self.keys, self.model, self.labels)

        
        
    def train_by_counting_4_states(self, data):
        """
        Computes training by counting with 4 states
        i, iMo, oMi, o
        uses self.train_by_counting
        :param data: a dict with the sequences
        :return:
        """

        self.labels = dict(i='i', o='o', iMo='M', oMi='M')

        for name in data:
            hiddens = list(data[name]['Z'])
            # modifying the hiddens to be 4 instead of 3
            for x in range(1, len(hiddens)):
                ## from inside to the membrane
                if hiddens[x - 1] == 'i' and hiddens[x] =='M':
                    hiddens[x] = 'iMo'
                ## from inside  still in  the membrane
                elif hiddens[x - 1] == 'iMo' and hiddens[x] =='M':
                    hiddens[x] = 'iMo'
                ## from outside to the membrane
                elif hiddens[x - 1] == 'o' and hiddens[x] =='M':
                    hiddens[x] = 'oMi'
                ## from outside still in  the membrane
                elif hiddens[x - 1] == 'oMi' and hiddens[x] =='M':
                    hiddens[x] = 'oMi'

                """
                ##### Already checked, nothing wrong, uncomment to check again
                ## from the oMi to outside shouldn't be possible
                ## just checking
                elif hiddens[x - 1] == 'oMi' and hiddens[x] =='o':
                    print "from oMi  to outside"

                ## from the iMo to inside shouldn't be possible
                ## just checking
                elif hiddens[x - 1] == 'iMo' and hiddens[x] =='i':
                    print "from iMo to inside"
                """

            # updating the input sequence to match the new states
            data[name]['Z'] = hiddens[:]
            ### At this step  we have each hidden sequences adapted with the new states
            ### so, why not use our first implementation to do the rest of the work??
        return self.train_by_counting(data)


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
