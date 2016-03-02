from utils import hmm, sequences
import os, math
from viterbi import Viterbi
from posterior import Posterior
from utils import outputs
from utils import compute_hmm
from utils import compare_tm_pred
from utils.sequences import Sequences

DATAFOLDER = "Training data"

KEYS = ['hidden', 'observables', 'pi', 'transitions', 'emissions']


if __name__ == '__main__':
    model = hmm.Model(KEYS)
    """
    ###STEP1###
    
    step1data = {}
    
    for i in range(9):
        #avoid problems with windows paths
        path = os.path.join(DATAFOLDER, "set160.%d.labels.txt"%i ) 
        seq_i = sequences.Sequences(path).sequences
        step1data.update(seq_i)
        
    model.train_by_counting_old(step1data)
    print model

    model.train_by_counting(step1data)
    print model
    
    ###STEP2###
    
    step2data = step1data

    model.train_by_counting_4_states(step2data)
    print model
    """
    ###step3###
    vit = Viterbi()
    scores = [0] * 10
    results = [None] * 10 #use this instead to output in fasta afterwards
    for i in range(10):
        step3data_train = {} ##reset data each time. If there is a way to update the old one it is likely bettter
        step3data_validate = {}
        step3data_validate = sequences.Sequences(os.path.join(DATAFOLDER, "set160.%d.labels.txt"%i))
        
        #train on all other than i
        for j in range(10):
            if (j!=i):
                path = os.path.join(DATAFOLDER, "set160.%d.labels.txt"%j ) 
                seq = sequences.Sequences(path).sequences
                step3data_train.update(seq)
                
        model.train_by_counting(step3data_train)
        
        
        #do viterbi prediction on set i
        for key, sequence in step3data_validate.get().items():
            ##                                true annotation         prediction
            scores[i] = compare_tm_pred.count(sequence['Z'],vit.decode(model, sequence['X'])[1])
        
        
        ##output results
        
        print "%d: %s"%(i, scores[i])
        compare_tm_pred.print_stats( *scores[i]  )
        
        
        ##bad score and it prints really weird sequences. I think maybe its supposed to be bad (its much better when you load the model from hmm-tm.txt from last week)
    
   ##need to either output to fasta to read in compare_tm, or just average over the things ^^^^
    ##I can do it but I want to be python-like.. 
    #look at the stats. Sequence 2 has no M predictions?
    
    
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