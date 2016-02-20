import numpy as np
import math

class Posterior:
    def __init__(self):
        self.z = []
        self.alpha = []
        self.beta = []


    def __logsum__(self, log_x, log_y):
        if log_x == -float('inf'): return log_y
        if log_y == -float('inf'): return log_x
        
        if log_x > log_y:
            return log_x + math.log( 1 + 2 ** (log_y - log_x))
        else:
            return log_y + math.log( 1 + 2 ** (log_x - log_y))

        
    def __alpha_recursion__(self, model, sequence):
        Z = len(model.hidden_states()) # num of hidden states
        X = len(sequence) # length of the input sequence
        # matrix[Z][X] full of -infinity
        self.alpha = np.full((Z,X), -float('inf'))
        self.beta = np.full((Z,X), -float('inf'))

        # fills the first column of w with pi(z0) * emission of the P(X[n] | Z)
        for k in range(Z): # where k is the index of each state
            self.alpha[k][0] = model.pi(k) + model.emission(k, model.index_observable(sequence[0]))

        # fills column by column, row by row
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
                            self.alpha[k][n] = self.__logsum__(self.alpha[k][n],self.alpha[j][n-1] 
                                                + model.transition(j, k))
                    
                    if self.alpha[k][n] != -float('inf'):
                        self.alpha[k][n] += model.emission(k, model.index_observable(sequence[n])) 
                  
                        
    def __beta_recursion__(self, model, sequence):
        
        """ def __bactracking__(self, model, sequence): BACKTRACK METHOD FROM VITERBI
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
        self.z = ''.join(self.z)"""
    
    def decode(self, model, sequence):

        self.__alpha_recursion__(model, sequence)
        print self.alpha
        #self.__beta_recursion__(model, sequence)
        #self.__bactracking__(model, sequence)

        #return ( np.max(self.w, axis=0)[-1], self.z )
