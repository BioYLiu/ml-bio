from utils import hmm
from viterbi import Viterbi
FILE = 'hmm-tm.txt'
KEYS = ['hidden', 'observables', 'pi', 'transitions', 'emissions']

model = hmm.Model(KEYS)
model.load(FILE)
print model
seq = 'MAKNLILWLVIAVVLMSVFQSF'
vit = Viterbi()
vit.decode(model, seq)