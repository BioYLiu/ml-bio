from utils import hmm, sequences
from viterbi import Viterbi
from posterior import Posterior
from utils import outputs

HMMFILE = 'hmm-tm.txt'
SEQUENCEFILE = 'sequences-project2.txt'
KEYS = ['hidden', 'observables', 'pi', 'transitions', 'emissions']

model = hmm.Model(KEYS)
model.load(HMMFILE)
sequences = sequences.Sequences(SEQUENCEFILE)
#print model
#print sequences
#model.viterbi(sequences['FTSH_ECOLI'])


vit = Viterbi()
post = Posterior()

probs = {}
seq = sequences.get()['FTSH_ECOLI']

states = post.decode(model, seq)

for key, value in sequences.get().items():
    probs[key] = post.decode(model, value)

outputs.to_project_1_sequences_file_from_posterior_decoding(sequences.get(), probs, 'posterior-decoding-sequences.txt')

"""
for key, value in sequences.get().items():
    probs[key] = vit.decode(model, value)


outputs.to_project_1_sequences_file(sequences.get(), probs, 'viterbi-sequences.txt')
outputs.to_project_1_probs_file(sequences.get(), probs, 'viterbi-probs.txt')
outputs.to_project_2_viterbi(sequences.get(), probs, 'viterbi-output.txt')

"""