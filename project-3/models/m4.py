def train(hmm, data):
    """
    Computes training by counting with 4 states
    i, iMo, oMi, o
    uses self.train_by_counting
    :param data: a dict with the sequences
    :return:
    """

    hmm.labels = dict(i='i', o='o', iMo='M', oMi='M')

    for name in data:
        hiddens = list(data[name]['Z'])
        # modifying the hiddens to be 4 instead of 3
        for x in range(1, len(hiddens)):
            ## from inside to the membrane
            if hiddens[x - 1] == 'i' and hiddens[x] =='M':
                hiddens[x] = 'iMo'
            ## from inside  still in  the membrane
            elif hiddens[x - 1] == 'iMo' and hiddens[x] =='M':
                hiddens[x] = 'iMo'
            ## from outside to the membrane
            elif hiddens[x - 1] == 'o' and hiddens[x] =='M':
                hiddens[x] = 'oMi'
            ## from outside still in  the membrane
            elif hiddens[x - 1] == 'oMi' and hiddens[x] =='M':
                hiddens[x] = 'oMi'

            
            ##### Already checked, nothing wrong, uncomment to check again
            ## from the oMi to outside shouldn't be possible
            ## just checking
            elif hiddens[x - 1] == 'oMi' and hiddens[x] =='o':
                print "from oMi  to outside"

            ## from the iMo to inside shouldn't be possible
            ## just checking
            elif hiddens[x - 1] == 'iMo' and hiddens[x] =='i':
                print "from iMo to inside"
            

        # updating the input sequence to match the new states
##        print "".join(data[name]['Z'])
##        print hiddens
        data[name]['Z'] = hiddens[:]
        ### At this step  we have each hidden sequences adapted with the new states
        ### so, why not use our first implementation to do the rest of the work??
    return hmm.train_by_counting(data)