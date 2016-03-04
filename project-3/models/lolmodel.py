def train(hmm, data):

    # i i i i M M M M M o o o o o M M M M M  i  i
    # 0 0 0 1 2 3 3 3 4 5 6 6 6 7 8 9 9 9 10 11 11
    hmm.labels = { '0': 'i', '1': 'i', '2': 'M', '3': 'M', '4': 'M', '5': 'o', '6': 'o',
        '7': 'o', '8': 'M', '9': 'M', '10': 'M', '11': 'i' }

    for name in data.keys():
        hiddens = list(data[name]['Z'])
        if hiddens[0] == 'i':
            hiddens[0] = '0'
        elif hiddens[0] == 'o':
            hiddens[0] = '6'

        for x in range(1,  len(hiddens)):
            if hiddens[x - 1] == '0':     # it was center i
                if hiddens[x] == 'i':   # now is i
                    hiddens[x] = '0'      # so still center i
                elif hiddens[x] == 'M': # now is M
                    hiddens[x - 1] = '1'  # so change previous to end i
                    hiddens[x] = '2'      # so now is init M
            elif hiddens[x - 1] == '2':   # it was init M
                if hiddens[x] == 'M':   # still M
                    hiddens[x] = '3'      # now is center M
            elif hiddens[x - 1] == '3':   # it was center M
                if hiddens[x] == 'M':   # still M
                    hiddens[x] = '3'      # so it is center M
                elif hiddens[x] == 'o': # now it is o
                    hiddens[x - 1] = '4'  # so previous was end M
                    hiddens[x] = '5'      # and now it is init o
            elif hiddens[x - 1] == '5':   # it was init o
                if hiddens[x] == 'o':   # still o
                    hiddens[x] = '6'      # now is center o
            elif hiddens[x - 1] == '6':   # it was center o
                if hiddens[x] == 'o':   # still o
                     hiddens[x] = '6'     # so still center o
                elif hiddens[x] == 'M': # now it is M
                    hiddens[x - 1] = '7'  # so change previous center o to end o
                    hiddens[x] = '8'      # so change current to init M
            elif hiddens[x - 1] == '8':   # it was init M
                if hiddens[x] == 'M':   # still M
                    hiddens[x] = '9'      # so center M
            elif hiddens[x - 1] == '9':   # it was center M
                if hiddens[x] == 'M':   # still M
                     hiddens[x] = '9'     # so center M
                elif hiddens[x] == 'i': # now it is i
                    hiddens[x - 1] = '10' # so previous was end M
                    hiddens[x] = '11'     # and now it is init i
            elif hiddens[x - 1] == '11':    # it was init i
                if hiddens[x] == 'i':   # still ti is i
                    hiddens[x] = '0'      # now is center i
            else:
                ## doesn't fit the model
                del data[name]
                break


        if name in data:
            data[name]['Z'] = hiddens[:]
        
    return hmm.train_by_counting(data)