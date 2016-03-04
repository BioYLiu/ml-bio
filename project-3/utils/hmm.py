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
    
    def __init__(self, keys, base_model=None, labels=None):
        self.keys = keys
        self.model = base_model or ''
        self.labels = labels
   

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
        #print self.model
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

                
                ##### Already checked, nothing wrong, uncomment to check again
                ## from the oMi to outside shouldn't be possible
                ## just checking
                elif hiddens[x - 1] == 'oMi' and hiddens[x] =='o':
                    print "from oMi  to outside"

                ## from the iMo to inside shouldn't be possible
                ## just checking
                elif hiddens[x - 1] == 'iMo' and hiddens[x] =='i':
                    print "from iMo to inside"
                

            # updating the input sequence to match the new states
    ##        print "".join(data[name]['Z'])
    ##        print hiddens
            data[name]['Z'] = hiddens[:]
            ### At this step  we have each hidden sequences adapted with the new states
            ### so, why not use our first implementation to do the rest of the work??
        return self.train_by_counting(data)

    
    def train_by_counting_first_and_last(self, data, number_of_aminoacids=1):
        # The idea could be define a variable number of aminoacids for 
        # the extrems of the helice, the default value could be 1, and 
        # test how it works
        # modifying the core core could be a little trickier, but not modifying
        # the extrems shouldn't be hard
        # And always keeping a core of 1 aminoacid, which can be repeated
        # so with a input parameter of 5 we would have the same model of the 
        # slide 20, with a 1 we would have some model with 8 states
        #
        
        #Then we can try training with 1...20 and compare AC's. Sounds a lot better than just trying 5
        #Agreed. Thats the plan. Gonna go see if they need me in the kitchen
        
        # yeah, but maybe we won't have so much improvement, reading what the slides say, but, we will know
        # because we will test it 
        #;-) COOl! ;-)
        # think so :-D
        #TODO
        
        self.labels = dict(i='i', o='o',
            iMo = 'M',   oMi = 'M',
            i1o = 'M',   o1i = 'M',
            i2o = 'M',   o2i = 'M',
            i3o = 'M',   o3i = 'M',
            i4o = 'M',   o4i = 'M',
            i5o = 'M',   o5i = 'M',
            i6o = 'M',   o6i = 'M',
            i7o = 'M',   o7i = 'M',
            i8o = 'M',   o8i = 'M',
            i9o = 'M',   o9i = 'M',
            i10o = 'M',   o10i = 'M')##This will only work for a fixed number of states. Just trying 5 real quick
                                    ##I maybe have a much nicer idea to try than this
                                    ##It makes the states as intended
        
        
        for name in data:
            hiddens = list(data[name]['Z'])
            for x in range(1, len(hiddens)): #forward pass
                ## from inside to the membrane
                if hiddens[x - 1] == 'i' and hiddens[x] =='M':
                    hiddens[x] = 'i1o'
                elif hiddens[x - 1] == 'i1o' and hiddens[x] =='M': ##this is fugly I know
                    hiddens[x] = 'i2o'
                elif hiddens[x - 1] == 'i2o' and hiddens[x] =='M':
                    hiddens[x] = 'i3o'
                elif hiddens[x - 1] == 'i3o' and hiddens[x] =='M':
                    hiddens[x] = 'i4o'
                elif hiddens[x - 1] == 'i4o' and hiddens[x] =='M':
                    hiddens[x] = 'i5o'
                elif hiddens[x - 1] == 'i5o' and hiddens[x] =='M':
                    hiddens[x] = 'iMo'
                elif hiddens[x - 1] == 'iMo' and hiddens[x] =='M': ##if in core just stay there
                    hiddens[x] = 'iMo'


                ## from outside to the membrane
                elif hiddens[x - 1] == 'o' and hiddens[x] =='M':
                    hiddens[x] = 'o1i'
                elif hiddens[x - 1] == 'o1i' and hiddens[x] =='M':
                    hiddens[x] = 'o2i'
                elif hiddens[x - 1] == 'o2i' and hiddens[x] =='M':
                    hiddens[x] = 'o3i'
                elif hiddens[x - 1] == 'o3i' and hiddens[x] =='M':
                    hiddens[x] = 'o4i'
                elif hiddens[x - 1] == 'o4i' and hiddens[x] =='M':
                    hiddens[x] = 'o5i'
                elif hiddens[x - 1] == 'o5i' and hiddens[x] =='M':
                    hiddens[x] = 'oMi'
                elif hiddens[x - 1] == 'oMi' and hiddens[x] =='M':
                    hiddens[x] = 'oMi'


            for x in range(len(hiddens)-2, -1, -1): #backward pass
                if hiddens[x + 1] == 'i' and hiddens[x] =='oMi': #It should be oMi now because of the forward pass
                    hiddens[x] = 'o10i'
                elif hiddens[x + 1] == 'o10i' and hiddens[x] =='oMi':
                    hiddens[x] = 'o9i'
                elif hiddens[x + 1] == 'o9i' and hiddens[x] =='oMi':
                    hiddens[x] = 'o8i'
                elif hiddens[x + 1] == 'o8i' and hiddens[x] =='oMi':
                    hiddens[x] = 'o7i'
                elif hiddens[x + 1] == 'o7i' and hiddens[x] =='oMi':
                    hiddens[x] = 'o6i'
                elif hiddens[x + 1] == 'o' and hiddens[x] =='iMo':
                    hiddens[x] = 'i10o'
                elif hiddens[x + 1] == 'i10o' and hiddens[x] =='iMo':
                    hiddens[x] = 'i9o'
                elif hiddens[x + 1] == 'i9o' and hiddens[x] =='iMo':
                    hiddens[x] = 'i8o'
                elif hiddens[x + 1] == 'i8o' and hiddens[x] =='iMo':
                    hiddens[x] = 'i7o'
                elif hiddens[x + 1] == 'i7o' and hiddens[x] =='iMo':
                    hiddens[x] = 'i6o'

            # print below to see translation to states
            #print data[name]['Z']
            #print hiddens[:]
            data[name]['Z'] = hiddens[:]
            
        return self.train_by_counting(data) 


    def train_by_lol_model(self, data):

        # i i i i M M M M M o o o o o M M M M M  i  i
        # 0 0 0 1 2 3 3 3 4 5 6 6 6 7 8 9 9 9 10 11 11
        self.labels = { '0': 'i', '1': 'i', '2': 'M', '3': 'M', '4': 'M', '5': 'o', '6': 'o',
            '7': 'o', '8': 'M', '9': 'M', '10': 'M', '11': 'i' }

        for name in data.keys():
            hiddens = list(data[name]['Z'])
            if hiddens[0] == 'i':
                hiddens[0] = '0'
            elif hiddens[0] == 'o':
                hiddens[0] = '6'

            for x in range(1,  len(hiddens)):
                if hiddens[x - 1] == '0':     # it was center i
                    if hiddens[x] == 'i':   # now is i
                        hiddens[x] = '0'      # so still center i
                    elif hiddens[x] == 'M': # now is M
                        hiddens[x - 1] = '1'  # so change previous to end i
                        hiddens[x] = '2'      # so now is init M
                elif hiddens[x - 1] == '2':   # it was init M
                    if hiddens[x] == 'M':   # still M
                        hiddens[x] = '3'      # now is center M
                elif hiddens[x - 1] == '3':   # it was center M
                    if hiddens[x] == 'M':   # still M
                        hiddens[x] = '3'      # so it is center M
                    elif hiddens[x] == 'o': # now it is o
                        hiddens[x - 1] = '4'  # so previous was end M
                        hiddens[x] = '5'      # and now it is init o
                elif hiddens[x - 1] == '5':   # it was init o
                    if hiddens[x] == 'o':   # still o
                        hiddens[x] = '6'      # now is center o
                elif hiddens[x - 1] == '6':   # it was center o
                    if hiddens[x] == 'o':   # still o
                         hiddens[x] = '6'     # so still center o
                    elif hiddens[x] == 'M': # now it is M
                        hiddens[x - 1] = '7'  # so change previous center o to end o
                        hiddens[x] = '8'      # so change current to init M
                elif hiddens[x - 1] == '8':   # it was init M
                    if hiddens[x] == 'M':   # still M
                        hiddens[x] = '9'      # so center M
                elif hiddens[x - 1] == '9':   # it was center M
                    if hiddens[x] == 'M':   # still M
                         hiddens[x] = '9'     # so center M
                    elif hiddens[x] == 'i': # now it is i
                        hiddens[x - 1] = '10' # so previous was end M
                        hiddens[x] = '11'     # and now it is init i
                elif hiddens[x - 1] == '11':    # it was init i
                    if hiddens[x] == 'i':   # still ti is i
                        hiddens[x] = '0'      # now is center i
                else:
                    ## doesn't fit the model
                    del data[name]
                    break


            if name in data:
                data[name]['Z'] = hiddens[:]
            
        return self.train_by_counting(data)
        
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
