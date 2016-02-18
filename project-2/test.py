from utils import hmm, sequences
HMMFILE = 'hmm-tm.txt'
SEQUENCEFILE = 'sequences-project2.txt'
KEYS = ['hidden', 'observables', 'pi', 'transitions', 'emissions']

model = hmm.Model(KEYS)
model.load(HMMFILE)
sequences = sequences.Sequences(SEQUENCEFILE)
print model
print sequences
#model.viterbi(sequences['FTSH_ECOLI'])