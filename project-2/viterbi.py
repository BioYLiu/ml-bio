import numpy as np
import math

class Viterbi:
    def __init__(self):
        self.w = []
        pass

    def __w_recursion__(self, model, sequence):
        Z = len(model.hidden_states())
        X = len(sequence)
        self.w = np.full((Z,X), -np.inf)

        for k in range(self.w):
            self.w[k][0] = math.log(model.pi(k)) + math.log(model.emission(k, model.index_observable(sequence[0])))

        for k in range(1, Z):
            for n in range(X):
                if model.emission(k, model.index_observable(X[n])) != 0:
                    for j in range(k):
                        #if self.w[]
                        pass

    def decode(self, model, sequence):

        self.__w_recursion__()



        print self.w
