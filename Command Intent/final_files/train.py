from docopt import docopt

def train_intent():
    pass

def train_ner():
    pass

def main():
    args = docopt(__doc__)
    print(args)

if __name__ == '__main__':
    main(docopt.docopt(__doc__))