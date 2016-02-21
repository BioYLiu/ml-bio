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
        self.alpha = np.full((Z, X), -float('inf'))

        # fills the first column of w with pi(z0) + emission of the P(X[n] | Z)
        for k in range(Z): # where k is the index of each state
            self.alpha[k][0] = model.pi(k) + model.emission(k, model.index_observable(sequence[0]))

        # fills column by column, row by row
        for n in range(1, X):
            for k in range(Z):
                # for each cell in each column
                # checks if the state k emitting the char of the sequence is higher than -infinity

                logsum = -float('inf')
                if model.emission(k, model.index_observable(sequence[n])) != -float('inf'):
                    # performs for each state
                    for j in range(Z):
                        # if exists the transition from j to k
                        if model.transition(j, k) != -float('inf'):
                            logsum = self.__logsum__(
                                                    logsum,
                                                    self.alpha[j][n - 1] +
                                                    model.transition(j, k)
                                                )
                    if logsum != -float('inf'):
                        logsum += model.emission(k, model.index_observable(sequence[n]))

                self.alpha[k][n] = logsum


    def __beta_recursion__(self, model, sequence):
        
        Z = len(model.hidden_states()) # num of hidden states
        X = len(sequence) # length of the input sequence
        # matrix[Z][X] full of -infinity
        self.beta = np.full((Z,X), -float('inf'))

        # fills the last column with 1
        for k in range(Z): # where k is the index of each state
            self.beta[k][-1] = 0.0

        # fills column by column, row by row
        for n in range(X-2, -1, -1):
            for k in range(Z):
                
                logsum = -float('inf')
                for j in range(Z):
                    logsum = self.__logsum__(
                                            logsum,
                                            self.beta[j][n + 1] +
                                            model.emission(j, model.index_observable(sequence[n + 1])) +
                                            model.transition(k, j)
                                        )
                    
                self.beta[k][n] = logsum
                

    def decode(self, model, sequence):
        
        X = len(sequence)
        Z = len(model.hidden_states())
        states = model.hidden_states()

        self.__alpha_recursion__(model, sequence)
        self.__beta_recursion__(model, sequence)

        """
        self.z = [None] * X
        for l in range (X):
            state = None
            bsf = -float('inf') #best so far
            for k in range(Z):
                contestor = self.alpha[k][l] + self.beta[k][l]
                if (contestor > bsf):
                    bsf = contestor
                    state = states[k]
            
            self.z[l] = state
        """
        ## foolchecks, surprisingly, these 2 lines below do the same as the above for in range(X)
        states = np.array(states)
        self.z = states[ np.argmax(self.alpha + self.beta, axis=0) ]

        self.z = ''.join(self.z)

        return self.z
