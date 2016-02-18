from utils import hmm
FILE = 'hmm-tm.txt'
KEYS = ['hidden', 'observables', 'pi', 'transitions', 'emissions']

model = hmm.Model(KEYS)
model.load(FILE)
print model