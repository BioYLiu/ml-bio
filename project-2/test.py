from utils import hmm, sequences
from viterbi import Viterbi
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

probs = {}


for key, value in sequences.get().items(): ##can we iterate over sequences and just output their strings? Instead of using keys
    probs[key] = vit.decode(model, value)
"""
    print ">%s"%(key)
    print "   %s"%(value)
    print "# %s"%(probs[key][1])##<--- in there is output from viterbibacktrack
    print
"""

outputs.to_project_1_sequences_file(sequences.get(), probs, 'viterbi-sequences.txt')
outputs.to_project_1_probs_file(sequences.get(), probs, 'viterbi-probs.txt')
outputs.to_project_2_viterbi(sequences.get(), probs, 'viterbi-output.txt')

