from utils import hmm
from utils import sequences as sequences_loader
from utils import merge_array_of_sequences as merge
import locale
from models import lolmodel, mx
import os, math
from viterbi import Viterbi
from posterior import Posterior
import numpy as np
from utils import compare_tm_pred
import sys
from utils import hmm_viterbi

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
    #model.train_by_counting_old(data) #I removed this old method. I have the code somewhere still
    model.train_by_counting(data)
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
    vit_total_ac = np.array([.0] * len(sequences))
    post_total_ac = np.array([.0] * len(sequences))
    vit = Viterbi()
    post = Posterior()
    

    for i in range(len(sequences)):
        vit_total_scores = np.zeros([4])
        post_total_scores = np.zeros([4])
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
            vit_pred_seq = vit.decode(model, sequence['X'])
            post_pred_seq = post.decode(model, sequence['X'])
            """
            print key
            print "PREDICTED"
            print pred_seq
            print "TRUE"
            print true_seq
            """
            tp, fp, tn, fn = compare_tm_pred.count(true_seq, vit_pred_seq)

            vit_total_scores += np.array([tp, fp, tn, fn])
            
            tp, fp, tn, fn = compare_tm_pred.count(true_seq, post_pred_seq)

            post_total_scores += np.array([tp, fp, tn, fn])
            if VERBOSE:
                print ">" + key
                compare_tm_pred.print_stats(tp, fp, tn, fn)
                print

        vit_total_ac[i] = compare_tm_pred.compute_stats(*vit_total_scores)[3]
        post_total_ac[i] = compare_tm_pred.compute_stats(*post_total_scores)[3]
        #print total_ac
        if VERBOSE:
            print "Summary 10-fold cross validation over index %i :"%(i)
          #  compare_tm_pred.print_stats( *total_scores  )
            print
            print
            print
            print "-------------------------------------------------------"
            if DEBUG:
                raw_input("press any key to continue\n")

    print "Overall viterbi result mean: %s, variance: %s"%(np.mean(vit_total_ac), np.var(vit_total_ac))
    print "Posterior mean: %s, variance %s"%(np.mean(post_total_ac), np.var(post_total_ac))




##this one loads them from the folder

def cross_validation_new(sequences, training_method, hmm, **kwargs):
    #DOES NOT TAKE DECODER ARG BUT PRINTS BOTH VIT AND POST SO WE DONT HAVE TO TRAIN TWICE
    """
    Performs the 10-fold cross-validation
    Requieres an array of dict sequences
    Requires the training function
    Requires a decoder objetct (Viterbi or Posterior)
    """
    # here we store the total_ac for each cross-validation
    vit_total_ac = np.array([.0] * len(sequences))
    post_total_ac = np.array([.0] * len(sequences))
    vit = Viterbi()
    post = Posterior()
    
    
    
    
    for i in range(len(sequences)):
        vit_total_scores = np.zeros([4])
        post_total_scores = np.zeros([4])
        # arrays with the sequences for training and for validation
        training_data_array = sequences[:]
        validation_data_array = [ training_data_array.pop(i) ]

        # merging the arrays into dictionaries
        training_data = merge(training_data_array)
        validation_data = merge(validation_data_array)
        # the training function returns a model
        mxacids = kwargs.get("mx_aminoacids")
        if (mxacids != None):
            model = training_method(hmm, training_data, kwargs.get("mx_aminoacids"))
        else:
            model = training_method(hmm, training_data)

        #do viterbi prediction on set i
        for key, sequence in validation_data.items():
            # the sequence from the file
            true_seq = sequence['Z']
            # the sequence decoded using viterbi, or posterior and the model generated
            vit_pred_seq = vit.decode(model, sequence['X'])
            post_pred_seq = post.decode(model, sequence['X'])
            """
            print key
            print "PREDICTED"
            print vit_pred_seq
            print "TRUE"
            print true_seq
            """
            tp, fp, tn, fn = compare_tm_pred.count(true_seq, vit_pred_seq)
            vit_total_scores += np.array([tp, fp, tn, fn])
            tp, fp, tn, fn = compare_tm_pred.count(true_seq, post_pred_seq)
            post_total_scores += np.array([tp, fp, tn, fn])

            if VERBOSE:
                print ">" + key
                compare_tm_pred.print_stats(tp, fp, tn, fn)
                print

        vit_total_ac[i] = compare_tm_pred.compute_stats(*vit_total_scores)[3]
        post_total_ac[i] = compare_tm_pred.compute_stats(*post_total_scores)[3]
        #print total_ac
        if VERBOSE:
            print "Summary 10-fold cross validation over index %i :"%(i)
            #compare_tm_pred.print_stats( *total_scores  )
            print
            print
            print
            print "-------------------------------------------------------"
            if DEBUG:
                raw_input("press any key to continue\n")

    
    print "Viterbi mean: %s, variance: %s"%(np.mean(vit_total_ac), np.var(vit_total_ac))
    print "Posterior mean: %s, variance %s"%(np.mean(post_total_ac), np.var(post_total_ac))

    
#    locale.setlocale(locale.LC_ALL, locale.normalize('da_DK'))
    #sys.stdout.write( "%s\t%s"%(np.mean(total_ac), np.var(total_ac))) ##output in sheet firnedly format)



if __name__ == '__main__':
    ###STEP1###
    # load the sequences ,  performs the training by counting and returns the model generated
##    step_1_sequences = load_sequences()
##    step_1_model = step_1(step_1_sequences)

    ###STEP2###
    # load the sequences ,  performs the training by counting and returns the model generated
##    step_2_sequences = load_sequences()
##    step_2_model = step_2(step_2_sequences)



#####   TEST EVERYTHING ####

    """
    model = hmm.Model(KEYS)
    sequences = load_sequences_as_array()

    print "Train by counting 3 states:"
#    print "Viterbi"
#    cross_validation(sequences, model.train_by_counting, Viterbi)
#    print "Posterior"
#    cross_validation(sequences, model.train_by_counting, Posterior)
    
    
    for x in range(6):
        print "Train by counting %d states:"%(4*(x+1))
#        cross_validation_new(sequences,mx.train,Viterbi, model, mx_aminoacids=x)
        cross_validation_new(sequences,mx.viterbitrain, model, mx_aminoacids=x)
#        cross_validation_new(sequences,mx.train,Posterior, model, mx_aminoacids=x)
        cross_validation_new(sequences,mx.viterbitrain, model, mx_aminoacids=x)
        print ""
    
    
#    print "Train by counting lol model:" 
#    print "Viterbi"
#    cross_validation_new(sequences, lolmodel.train, Viterbi, model) 
#    print "Posterior"
#    cross_validation_new(sequences, lolmodel.train, Posterior, model) 



    """
   
    """     TEST RESULTS Training by counting
    
            Train by counting 3 states:
            Viterbi
            Overall result mean: 0.676441765877, variance: 0.00434450326705
            Posterior
            Overall result mean: 0.787618369207, variance: 0.00128948356058
            Train by counting 4 states:
            Viterbi
            Overall result mean: 0.680680926674, variance: 0.00483538599374
            Posterior
            Overall result mean: 0.780762655675, variance: 0.0018264024132
            Train by counting 8 states:
            Viterbi
            Overall result mean: 0.690670674091, variance: 0.00393775000278
            Posterior
            Overall result mean: 0.771872092805, variance: 0.00185616981759
            Train by counting 12 states:
            Viterbi
            Overall result mean: 0.702524411917, variance: 0.00322806235048
            Posterior
            Overall result mean: 0.76668613816, variance: 0.00161741834358
            Train by counting 16 states:
            Viterbi
            Overall result mean: 0.714631621683, variance: 0.00440273036113
            Posterior
            Overall result mean: 0.764515427626, variance: 0.00166258473196
            Train by counting 20 states:
            Viterbi
            Overall result mean: 0.711587192249, variance: 0.00409613914721
            Posterior
            Overall result mean: 0.760310775588, variance: 0.00159581113985
            Train by counting 24 states:
            Viterbi
            Overall result mean: 0.726542923523, variance: 0.00365988474218
            Posterior
            Overall result mean: 0.758140263941, variance: 0.00140248186867
            Train by counting lol model:
            Viterbi
            Overall result mean: 0.727260403909, variance: 0.00308761518303
            Posterior
            Overall result mean: 0.789227688282, variance: 0.00157365490291
            
    """
    sequences = load_sequences()
    viterbi_model  = hmm_viterbi.Model(KEYS)
    viterbi_model.load('initial_model.txt')
    print viterbi_model
    print "training"
    viterbi_model.train(sequences, 10)
    

    
    

    
    
    
    
    
