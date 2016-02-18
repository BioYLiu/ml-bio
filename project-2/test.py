from utils import hmm, sequences
from viterbi import Viterbi
HMMFILE = 'hmm-tm.txt'
SEQUENCEFILE = 'sequences-project2.txt'
KEYS = ['hidden', 'observables', 'pi', 'transitions', 'emissions']

model = hmm.Model(KEYS)
model.load(HMMFILE)
sequences = sequences.Sequences(SEQUENCEFILE)
print model
print sequences
#model.viterbi(sequences['FTSH_ECOLI'])

seq = 'MAKNLILWLVIAVVLMSVFQSF'
vit = Viterbi()
vit.decode(model, seq)

