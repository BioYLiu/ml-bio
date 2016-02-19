
class Sequences:
    def __init__(self, filename):
        self.filename = filename
        self.load(filename)

    def load(self, filename):
        data = {}
        # just fetch the lines which are not empty
        with open(filename, 'r') as f:
            lines = f.readlines()
            x = 0
            while x < len(lines):
                while lines[x].strip() == "":##skips whitespace lines in case someone is lazy with the input form
                    x+=1
                name = lines[x].strip('\n').lstrip('>')
                x+=1
                while lines[x].strip() == "":
                    x+=1
                emissions = lines[x].strip('\n').lstrip('  ')
                x+=1
                data[name] = emissions
                
        self.sequences = data

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