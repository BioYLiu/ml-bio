from utils import hmm, sequences
from viterbi import Viterbi
from posterior import Posterior
from utils import outputs
from utils import compute_hmm
from utils.sequences import Sequences

DATAFOLDER = "Training data/"

KEYS = ['hidden', 'observables', 'pi', 'transitions', 'emissions']


if __name__ == '__main__':
    model = hmm.Model(KEYS)
    
    ###STEP1###
    
    step1data = {}
    
    for i in range(9):
        seq_i = sequences.Sequences(DATAFOLDER+"set160.%d.labels.txt"%i).sequences
        step1data.update(seq_i)
        
    #model.train_by_counting(step1data)
    
    ###STEP2###
    
    step2data = step1data
    
    ###########
    print model
    
    
    
    
    
    
    
    
    
    
    
    
    # load methods
   # vit = Viterbi()
    #post = Posterior()
    
    """

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
    """