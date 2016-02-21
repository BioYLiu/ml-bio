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

        # fills the first column of w with pi(z0) * emission of the P(X[n] | Z)
        for k in range(Z): # where k is the index of each state
            self.alpha[k][0] = model.pi(k) + model.emission(k, model.index_observable(sequence[0]))

        # fills column by column, row by row
        for n in range(1, X):
            for k in range(Z):
                # for each cell in each column
                # checks if the state k emmitting the char of the sequence is higher than -infinity
                
                """ MAYBE change below (rest of method) to be more like pseudocode to remove potentiel errors  """
                logsum = -float('inf')
                if model.emission(k, model.index_observable(sequence[n])) != -float('inf'):
                    # performs for each state

                    ### moving to a similar pseudocode

                    for j in range(Z):
                        # if exists the transition from j to k
                        if model.transition(j, k) != -float('inf'):
                            logsum = self.__logsum__(
                                                    logsum,
                                                    self.alpha[j][n - 1] + model.transition(j, k)
                                                )
                    if logsum != -float('inf'):
                        logsum += model.emission(k, model.index_observable(sequence[n]))

                self.alpha[k][n] = logsum

                """
                if model.emission(k, model.index_observable(sequence[n])) != -float('inf'):
                    for j in range(Z):
                        if model.transition(j, k) != -float('inf'):
                            # computes the log_sum of the current cell and the previous state +  the transition from
                            # state j to state k
                            self.alpha[k][n] = self.__logsum__(
                                                    self.alpha[k][n], 
                                                    self.alpha[j][n-1] 
                                                    + model.transition(j, k)
                                                )
                    
                    # as we are in the log space and computing sums, we need to check if the 
                    # value of the current cell is different to -inf, if it is, we can add the sum
                    # of the emission, if it was -inf we couldn't add anything
                    if self.alpha[k][n] != -float('inf'):
                        self.alpha[k][n] += model.emission(k, model.index_observable(sequence[n])) 
                  """
                        
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
                
                lsum = -float('inf')
                for j in range(Z):
                    """
                    if model.transition(k, j) != -float('inf'):
                       # gets the maximum value between the current value in w[k][n]
                       # the state k emmitting the char of the sequence plus
                        # the value in w[k][n-1] plus the transition from j to k
                        
                        
                        
                        
                        self.beta[k][n] = self.__logsum__( 
                                            self.beta[k][n], 
                                            self.beta[j][n + 1] 
                                            + model.transition(k, j)
                                            + model.emission(j, model.index_observable(sequence[n + 1])) 
                                        )
                    """
                    
                    
                    #more foolproof version attempt (wait thats more or less the same)
                    
                    lsum = self.__logsum__(lsum, self.beta[j][n+1]+
                                            model.emission(j, model.index_observable(sequence[n+1]))+
                                            model.transition(k,j))
                    
                self.beta[k][n] = lsum      
                
                
            
                                            
                    
                    
                    
    def decode(self, model, sequence):
        
        X = len(sequence)
        Z = len(model.hidden_states())
        states = model.hidden_states()
        self.z = [None] * X
        self.__alpha_recursion__(model, sequence)
        self.__beta_recursion__(model, sequence)
        #print self.alpha
        #print self.beta
        px = reduce( self.__logsum__, [ self.alpha[x][-1] for x in range(Z) ], -float('inf') )
        """
        
        px = -float('inf')
        for l in range(0,Z):
            px = self.__logsum__(px, self.alpha[l][-1])
        """    
        
        for l in range (X):
            state = None
            bsf = -float('inf') #best so far
            for k in range(Z):
                contestor = self.alpha[k][l] + self.beta[k][l]
                if (contestor > bsf):
                    bsf = contestor
                    state = states[k]
            
            self.z[l] = state
            
        self.z =  ''.join(self.z)
        return self.z
