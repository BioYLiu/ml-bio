
class Sequences:
    def __init__(self):
        pass

    def load_sequences(self, filename):
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
                data[name] = dict(X=emissions, Z=hiddens, name=name)
        self.sequences = data

    def __str__(self):
        return str(self.sequences)