import numpy as np
import math

class Viterbi:
    def __init__(self):
        self.w = []
        self.z = []


    def __w_recursion__(self, model, sequence):
        Z = len(model.hidden_states()) # num of hidden states
        X = len(sequence) # length of the input sequence
        # matrix[Z][X] full of -infinity
        self.w = np.full((Z,X), -float('inf'))

        # fills the first column of w with pi(z0) * emission of the P(X[n] | Z)
        for k in range(Z): # where k is the index of each state
            self.w[k][0] = model.pi(k) + model.emission(k, model.index_observable(sequence[0]))
        
        # fills column by column, row by row
        # starts b 1 because the first element of the input is the PI
        for n in range(1, X):
            for k in range(Z):
                # for each cell in each column
                # checks if the state k emmitting the char of the sequence is higher than -infinity
                if model.emission(k, model.index_observable(sequence[n])) != -float('inf'):
                    # performs for each state
                    for j in range(Z):
                        # if exists the transition from j to k
                        if model.transition(j, k) != -float('inf'):
                            # gets the maximum value between the current value in w[k][n]
                            # the state k emmitting the char of the sequence plus
                            # the value in w[k][n-1] plus the transition from j to k
                            self.w[k][n] = max(
                                                self.w[k][n], 
                                                model.emission(k, model.index_observable(sequence[n])) 
                                                + self.w[j][n-1] 
                                                + model.transition(j, k)
                                            )
                        
    
    def __bactracking__(self, model, sequence):
        hidden_states = model.hidden_states()
        Z = len(sequence) # length of the input sequence
        #self.z = np.full(Z, None)
        self.z = [None] * Z
        # gets the index of the maxim argument in the matrix z in 
        # axis 0( which means column by column) and picks the last one
        # so then picks the hidden state based in the index
        self.z[-1] = hidden_states[np.argmax(self.w, axis=0)[-1]]
        # from Z-1 to 0
        for n in range(Z-2, -1, -1):
            indexes = []
            # now computes for each column
            # the sum of state picked in the previous iteration emitting the character of the previous iteration plus
            # the value in w[k][n] plus
            # the transition from the state k to the state from the previous iteration
            for k in range(len(hidden_states)):
                indexes.append( 
                                model.emission( model.index_hidden_state(self.z[n + 1]), model.index_observable(sequence[n + 1]))
                                + self.w[k][n]
                                + model.transition(k, model.index_hidden_state(self.z[n+1] ) )
                            )
            self.z[n] = hidden_states[np.argmax(indexes)]
            
        # converts the array of chars to a string
        # adjustment, now we need to check if we have labels (i.e: iMo instead of M), to generate the correct
        # output sequence from the
        if model.get_labels()is not None:
            self.z = model.translate_hidden_states(self.z)
        else:
            self.z = ''.join(self.z)
    
    def decode(self, model, sequence):

        self.__w_recursion__(model, sequence)
        self.__bactracking__(model, sequence)

        return ( np.max(self.w, axis=0)[-1], self.z )
        
        
