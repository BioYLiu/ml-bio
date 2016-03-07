from viterbi import Viterbi

def train(hmm, data, number_of_aminoacids):
    # The idea could be define a variable number of aminoacids for 
    # the extrems of the helice, the default value could be 1, and 
    # test how it works
    # modifying the core core could be a little trickier, but not modifying
    # the extrems shouldn't be hard
    # And always keeping a core of 1 aminoacid, which can be repeated
    # so with a input parameter of 5 we would have the same model of the 
    # slide 20, with a 1 we would have some model with 8 states
    #
    
    #Then we can try training with 1...20 and compare AC's. Sounds a lot better than just trying 5
    #Agreed. Thats the plan. Gonna go see if they need me in the kitchen
    
    # yeah, but maybe we won't have so much improvement, reading what the slides say, but, we will know
    # because we will test it 
    #;-) COOl! ;-)
    # think so :-D
    #TODO
    mino = number_of_aminoacids
    hmm.labels = dict(i='i', o='o',
        iMo = 'M',   oMi = 'M')

    if mino > 0:
        hmm.labels.update(i1o = 'M',   o1i = 'M',
                            i10o = 'M',   o10i = 'M')
    if mino > 1:
        hmm.labels.update(i2o = 'M',   o2i = 'M',
                            i9o = 'M',   o9i = 'M')
    if mino > 2:
        hmm.labels.update(i3o = 'M',   o3i = 'M',
                            i8o = 'M',   o8i = 'M')
    if mino > 3:
         hmm.labels.update(i4o = 'M',   o4i = 'M',
                            i7o = 'M',   o7i = 'M')
    if mino > 4:
        hmm.labels.update(i5o = 'M',   o5i = 'M',
                            i6o = 'M',   o6i = 'M')
    
    ##This will only work for a fixed number of states. Just trying 5 real quick
                            ##I maybe have a much nicer idea to try than this
                            ##It makes the states as intended

    
    for name in data:
        hiddens = list(data[name]['Z'])
        for x in range(1, len(hiddens)): #forward pass
            ## from inside to the membrane
            if hiddens[x - 1] == 'i' and hiddens[x] =='M':
                if mino>0:
                    hiddens[x] = 'i1o'
                else:
                    hiddens[x] = 'iMo'
            elif hiddens[x - 1] == 'i1o' and hiddens[x] =='M':
                if mino>1:
                    hiddens[x] = 'i2o'
                else:
                    hiddens[x] = 'iMo'
            elif hiddens[x - 1] == 'i2o' and hiddens[x] =='M':
                if mino>2:
                    hiddens[x] = 'i3o'
                else:
                    hiddens[x] = 'iMo'
            elif hiddens[x - 1] == 'i3o' and hiddens[x] =='M':
                if mino>3:
                    hiddens[x] = 'i4o'
                else:
                    hiddens[x] = 'iMo'
            elif hiddens[x - 1] == 'i4o' and hiddens[x] =='M':
                if mino>4:
                    hiddens[x] = 'i5o'
                else:
                    hiddens[x] = 'iMo'
            elif hiddens[x - 1] == 'i5o' and hiddens[x] =='M':
                    hiddens[x] = 'iMo'
            elif hiddens[x - 1] == 'iMo' and hiddens[x] =='M':
                    hiddens[x] = 'iMo'
            elif hiddens[x - 1] == 'i2o' and hiddens[x] =='M' and mino>2: ## we can remove all the "and hiddens[x]" by checking it first"
                hiddens[x] = 'i3o'
            elif hiddens[x - 1] == 'i3o' and hiddens[x] =='M' and mino>3:
                hiddens[x] = 'i4o'
            elif hiddens[x - 1] == 'i4o' and hiddens[x] =='M' and mino>4: ##if 5 or larger.. its just 5
                hiddens[x] = 'i5o'
            elif hiddens[x - 1] == 'i5o' and hiddens[x] =='M':
                hiddens[x] = 'iMo'
            elif hiddens[x - 1] == 'iMo' and hiddens[x] =='M': ##if in core just stay there
                hiddens[x] = 'iMo'
            ## from outside to the membrane
            elif hiddens[x - 1] == 'o' and hiddens[x] =='M':
                if mino>0:
                    hiddens[x] = 'o1i'
                else:
                    hiddens[x] = 'oMi'
            elif hiddens[x - 1] == 'o1i' and hiddens[x] =='M':
                if mino>1:
                    hiddens[x] = 'o2i'
                else:
                    hiddens[x] = 'oMi'
            elif hiddens[x - 1] == 'o2i' and hiddens[x] =='M':
                if mino>2:
                    hiddens[x] = 'o3i'
                else:
                    hiddens[x] = 'oMi'
            elif hiddens[x - 1] == 'o3i' and hiddens[x] =='M':
                if mino>3:
                    hiddens[x] = 'o4i'
                else:
                    hiddens[x] = 'oMi'
            elif hiddens[x - 1] == 'o4i' and hiddens[x] =='M':
                if mino>4:
                    hiddens[x] = 'o5i'
                else:
                    hiddens[x] = 'oMi'
            elif hiddens[x - 1] == 'o5i' and hiddens[x] =='M':
                hiddens[x] = 'oMi'
            elif hiddens[x - 1] == 'oMi' and hiddens[x] =='M':
                hiddens[x] = 'oMi'


        for x in range(len(hiddens)-2, -1, -1): #backward pass
            if hiddens[x + 1] == 'i' and hiddens[x] =='oMi':
                if mino>0:
                    hiddens[x] = 'o10i'
                else:
                    hiddens[x] = 'oMi'

            elif hiddens[x + 1] == 'o10i' and hiddens[x] =='oMi':
                if mino>1:
                    hiddens[x] = 'o9i'
                else:
                    hiddens[x] = 'oMi'

            elif hiddens[x + 1] == 'o9i' and hiddens[x] =='oMi':
                if mino>2:
                    hiddens[x] = 'o8i'
                else:
                    hiddens[x] = 'oMi'

            elif hiddens[x + 1] == 'o8i' and hiddens[x] =='oMi':
                if mino>3:
                    hiddens[x] = 'o7i'
                else:
                    hiddens[x] = 'oMi'

            elif hiddens[x + 1] == 'o7i' and hiddens[x] =='oMi':
                if mino>4:
                    hiddens[x] = 'o6i'
                else:
                    hiddens[x] = 'oMi'
                
          
            elif hiddens[x + 1] == 'o' and hiddens[x] =='iMo':
                if mino>0:
                    hiddens[x] = 'i10o'
                else:
                    hiddens[x] = 'iMo'
            elif hiddens[x + 1] == 'i10o' and hiddens[x] =='iMo':
                if mino>1:
                    hiddens[x] = 'i9o'
                else:
                    hiddens[x] = 'iMo'
            elif hiddens[x + 1] == 'i9o' and hiddens[x] =='iMo':
                if mino>2:
                    hiddens[x] = 'i8o'
                else:
                    hiddens[x] = 'iMo'
            elif hiddens[x + 1] == 'i8o' and hiddens[x] =='iMo':
                if mino>3:
                    hiddens[x] = 'i7o'
                else:
                    hiddens[x] = 'iMo'
            elif hiddens[x + 1] == 'i7o' and hiddens[x] =='iMo':
                if mino>4:
                    hiddens[x] = 'i6o'
                else: 
                    hiddens[x] = 'iMo'

        # print below to see translation to states
##        print data[name]['Z']
##        print hiddens[:]
        data[name]['Z'] = hiddens[:]
        
    return hmm.train_by_counting(data) 
"""    
def viterbitrain(hmm, datainp, number_of_aminoacids):
    iterations = 4
    data = datainp #make a copy?
    vit = Viterbi()
    curr_model = None
    for i in range(iterations): #other stopping criterion could be better
        curr_model = train(hmm, data, number_of_aminoacids)
        for name in data:
            data[name]['Z'] = vit.decode(curr_model, data[name]['X'])
    
    return curr_model
"""
def viterbitrain(hmm, data, number_of_aminoacids):
    iterations = 4
    vit=Viterbi()
    
    