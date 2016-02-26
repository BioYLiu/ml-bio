class Sequences:
    def __init__(self, filename):
        self.filename = filename
        self.sequences = self.load_sequences(filename)

    def load_sequences(self, filename): #Taken from project1
        data = {}
        # just fetch the lines which are not empty
        with open(filename, 'r') as f:
            lines = f.readlines()
            x = 0
            while x < len(lines):
                name = lines[x].strip('\n').lstrip('>')
                x += 1
                emissions = lines[x].strip('\n').lstrip(' ')
                x += 1
                hiddens = lines[x].strip('\n').lstrip('# ')
                x += 2
                data[name] = dict(X=emissions, Z=hiddens)
        return data;

    def __str__(self):
        return str(self.sequences)
        
    def __len__(self):
        return len(self.sequences)
    
    def get(self):
        return self.sequences
     
    def get_by_key(self, key):
        return self.sequences[key]
        
if __name__ == "__main__":
    
    import sequences
    SEQUENCEFILE = '../sequences-project2.txt'
    sequences = sequences.Sequences(SEQUENCEFILE)
    print sequences