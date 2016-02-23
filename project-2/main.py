from utils import hmm, sequences
from viterbi import Viterbi
from posterior import Posterior
from utils import outputs
from utils import compute_hmm

HMMFILE = 'hmm-tm.txt'

SEQUENCEFILE = 'test-sequences-project2.txt'
#SEQUENCEFILE = '../../sequences-project2.txt' #Two folders up!!! :-P #nice

KEYS = ['hidden', 'observables', 'pi', 'transitions', 'emissions']


"""


#seq = sequences.get()['FTSH_ECOLI']

#states = post.decode(model, seq)



#outputs.to_project_1_sequences_file_from_posterior_decoding(sequences.get(), probs, 'posterior-decoding-sequences.txt')

outputs.to_project_1_sequences_file(sequences.get(), probs, 'viterbi-sequences.txt')
outputs.to_project_1_probs_file(sequences.get(), probs, 'viterbi-probs.txt')


"""
if __name__ == '__main__':
    model = hmm.Model(KEYS)
    model.load(HMMFILE)
    sequences = sequences.Sequences(SEQUENCEFILE)
    # load methods
    vit = Viterbi()
    post = Posterior()
    
    

    # viterbi
    probs = {}
    for key, sequence in sequences.get().items():
        probs[key] = vit.decode(model, sequence)

    outputs.to_project_2_viterbi(sequences.get(), probs, 'pred-test-sequences-project2-viterbi.txt')

    probs = {}
    for key, value in sequences.get().items():
        sequence = {
            'Z': post.decode(model, value),
            'X': value
        }
        log_joint = compute_hmm(model, sequence)

        probs[key] = ( log_joint, sequence['Z'])

    #outputs.to_project_2_posterior(sequences.get(), probs, 'posterior-output.txt')
    #outputs.to_project_2_posterior(sequences.get(), probs, 'pred-test-sequences-project2-posterior.txt')
    # testing
    #probs = { key: value[1] for key, value in probs.items() }
    #outputs.to_project_1_sequences_file_from_posterior_decoding(sequences.get(), probs, 'posterior-decoding-sequences.txt')