from utils import hmm
from utils import sequences as sequences_loader
from utils import merge_array_of_sequences as merge
import os, math
from viterbi import Viterbi
from posterior import Posterior
import numpy as np
from utils import compare_tm_pred

DATAFOLDER = "Training data"

KEYS = ['hidden', 'observables', 'pi', 'transitions', 'emissions']
DEBUG = False
VERBOSE = False

def load_sequences():
    """
    Load all the sequences in file DATAFOLDER
    :return: A dictionary with all the sequences
    """
    sequences_dict = {}
    for filename in os.listdir(DATAFOLDER):
        #avoid problems with windows paths
        path = os.path.join(DATAFOLDER, filename )
        seq_i = sequences_loader.Sequences(path).sequences
        sequences_dict.update(seq_i)

    return sequences_dict

def load_sequences_as_array():
    """
    Load all the sequences in file DATAFOLDER
    :return: An array with a dict of sequences in each index
    """
    sequences_array = []
    for filename in os.listdir(DATAFOLDER):
        #avoid problems with windows paths
        path = os.path.join(DATAFOLDER, filename)
        seq_i = sequences_loader.Sequences(path).sequences
        sequences_array.append(seq_i)

    return sequences_array

def step_1(data):
    model = hmm.Model(KEYS)
    #model.train_by_counting_old(data)
    model.train_by_counting_old(data)
    return model

def step_2(data):
    model = hmm.Model(KEYS)
    model.train_by_counting_4_states(data)
    return model


def cross_validation(sequences, training_method, decoder):
    """
    Performs the 10-fold cross-validation
    Requieres an array of dict sequences
    Requires the training function
    Requires a decoder objetct (Viterbi or Posterior)
    """
    # here we store the total_ac for each cross-validation
    total_ac = np.array([.0] * len(sequences))
    
    
    dec = decoder()

    for i in range(len(sequences)):
        total_scores = np.zeros([4])
        # arrays with the sequences for training and for validation
        training_data_array = sequences[:]
        validation_data_array = [ training_data_array.pop(i) ]

        # merging the arrays into dictionaries
        training_data = merge(training_data_array)
        validation_data = merge(validation_data_array)
        # the training function returns a model
        model = training_method(training_data)

        #do viterbi prediction on set i
        for key, sequence in validation_data.items():
            # the sequence from the file
            true_seq = sequence['Z']
            # the sequence decoded using viterbi, or posterior and the model generated
            pred_seq = dec.decode(model, sequence['X'])
            """
            print key
            print "PREDICTED"
            print pred_seq
            print "TRUE"
            print true_seq
            """
            tp, fp, tn, fn = compare_tm_pred.count(true_seq, pred_seq)

            total_scores += np.array([tp, fp, tn, fn])

            if VERBOSE:
                print ">" + key
                compare_tm_pred.print_stats(tp, fp, tn, fn)
                print

        total_ac[i] = compare_tm_pred.compute_stats(*total_scores)[3]
        #print total_ac
        if VERBOSE:
            print "Summary 10-fold cross validation over index %i :"%(i)
            compare_tm_pred.print_stats( *total_scores  )
            print
            print
            print
            print "-------------------------------------------------------"
            if DEBUG:
                raw_input("press any key to continue\n")

    print "Overall result mean: %s, variance: %s"%(np.mean(total_ac), np.var(total_ac))

if __name__ == '__main__':


    ###STEP1###
    # load the sequences ,  performs the training by counting and returns the model generated
##    step_1_sequences = load_sequences()
##    step_1_model = step_1(step_1_sequences)

    ###STEP2###
    # load the sequences ,  performs the training by counting and returns the model generated
##    step_2_sequences = load_sequences()
##    step_2_model = step_2(step_2_sequences)

    sequences = load_sequences_as_array()
    
    ###STEP3###
    print "Step 3 - Train by counting with Viterbi"
    #Viterbi  mean: 0.676441765877, variance: 0.00434450326705
    model = hmm.Model(KEYS)
    sequences = load_sequences_as_array()
    #Viterbi  mean: 0.676441765877, variance: 0.00434450326705
    cross_validation(sequences, model.train_by_counting, Viterbi)


    ###STEP4###
    print "Step 4 - Train by counting 4 states with Viterbi"
    #Viterbi mean: 0.680680926674, variance: 0.00483538599374
    model = hmm.Model(KEYS)
    cross_validation(sequences, model.train_by_counting_4_states, Viterbi)

    ##STEP5##
    print "Step 5.3 - Train by counting with Posterior"
    #post     mean: 0.787618369207, variance: 0.00128948356058
    model = hmm.Model(KEYS)
    cross_validation(sequences, model.train_by_counting, Posterior)

    print "Step 5.4 - Train by counting four states with Posterior"
    #post    mean: 0.780762655675, variance: 0.0018264024132
    model = hmm.Model(KEYS)
    cross_validation(sequences, model.train_by_counting_4_states, Posterior)

    print "Step 6.1 - Train by counting 24 states with Viterbi"
    #Viterbi, 24-states mean: 0.726542923523, variance: 0.00365988474218
    model = hmm.Model(KEYS)
    cross_validation(sequences, model.train_by_counting_first_and_last, Viterbi) 
    
    print "Step 6.2 - Train by counting 24 states with Posterior"
    #Viterbi, 24-states mean: 0.726542923523, variance: 0.00365988474218
    model = hmm.Model(KEYS)
    cross_validation(sequences, model.train_by_counting_first_and_last, Posterior) 
    """
    ### lol model, sounds better
    # i i i i M M M M M o o o o o M M M M M  i  i i
    # 0 0 0 1 2 3 3 3 4 5 6 6 6 7 8 9 9 9 10 11 0 0
    """
    print "Step 6.3 - Train by counting  lol_model with Viterbi" 
    #Viterbi, 24-states mean: 0.726542923523, variance: 0.00365988474218
    model = hmm.Model(KEYS)
    cross_validation(sequences, model.train_by_lol_model, Viterbi) 

    print "Step 6.4 - Train by counting  lol_model with Posterior" 
    #Viterbi, 24-states mean: 0.726542923523, variance: 0.00365988474218
    model = hmm.Model(KEYS)
    cross_validation(sequences, model.train_by_lol_model, Posterior) 
    
    
    

    """

    ###step3###
    vit = Viterbi()
    scores = [0] * 10
    results = [None] * 10 #use this instead to output in fasta afterwards
    for i in range(10):
        step3data_train = {} ##reset data each time. If there is a way to update the old one it is likely bettter
        step3data_validate = {}
        step3data_validate = sequences_loader.Sequences(os.path.join(DATAFOLDER, "set160.%d.labels.txt"%i))
        
        #train on all other than i
        for j in range(10):
            if (j!=i):
                path = os.path.join(DATAFOLDER, "set160.%d.labels.txt"%j ) 
                seq = sequences_loader.Sequences(path).sequences
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