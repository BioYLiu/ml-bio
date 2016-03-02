
def to_project_1_sequences_file(sequences, probs, filename):
    with open(filename, 'w') as f:
        for key, value in sequences.items():
            f.write(">%s\n"%(key))
            f.write("   %s\n"%(value))
            f.write("# %s\n"%(probs[key][1]))
            f.write("\n")


def to_project_1_sequences_file_from_posterior_decoding(sequences, probs, filename):
    with open(filename, 'w') as f:
        for key, value in sequences.items():
            f.write(">%s\n"%(key))
            f.write("   %s\n"%(value))
            f.write("# %s\n"%(probs[key]))
            f.write("\n")


def to_project_1_probs_file(sequences, probs, filename):
    with open(filename, 'w') as f:
        for key, value in sequences.items():
            f.write("%s\n"%(key))
            f.write("log P(x,z) =   %s\n"%(probs[key][0]))


def to_project_2_viterbi(sequences, probs, filename,  key_order = None):

    if key_order is not None:
        keys = key_order
    else:
        keys = sequences.keys()

    with open(filename, 'w') as f:
        f.write("; Viterbi-decodings of sequences-project2.txt using HMM hmm-tm.txt by Martin and Juan\n")
        f.write("\n")

        for key in keys:
            if key in sequences:
                f.write(">%s\n"%(key))
                f.write("%s\n"%(sequences[key]))
                f.write("#\n")
                f.write("%s\n"%(probs[key][1]))
                f.write("; log P(x,z) =   %s\n"%(probs[key][0]))
                f.write("\n")

def to_project_2_posterior(sequences, probs, filename, key_order = None):

    if key_order is not None:
        keys = key_order
    else:
        keys = sequences.keys()

    with open(filename, 'w') as f:
        f.write("; Posterior-decodings of sequences-project2.txt using HMM hmm-tm.txt by Martin and Juan\n")
        f.write("\n")
        for key in keys:
            if key in sequences:
                f.write(">%s\n"%(key))
                f.write("%s\n"%(sequences[key]))
                f.write("#\n")
                f.write("%s\n"%(probs[key][1]))
                f.write("; log P(x,z) = %s\n"%(probs[key][0]))
                f.write("\n")
                
