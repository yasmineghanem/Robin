"""
Usage:
    main.py train_intent data_path [options] 
    main.py test_intent data_path [options]
    main.py train_ner data_path [options]
    main.py test_ner data_path [options] 

Options:
    --epochs=<int>              Number of epochs [default: 10]
    --lr=<float>                Learning rate [default: 0.01]
    --dropout-rate=<float>      Dropout rate [default: 0.4]
    --model-path=<str>          Path to save model [default: ../models/model.h5]
    --embedding-dim=<int>       Embedding dimension [default: 128]
    --lstm-units=<int>          LSTM units [default: 64]
"""

'''
    This is the main file that will be used to train and test the intent detection and named entity recognition models.
'''

from docopt import docopt

def main():
    args = docopt(__doc__)
    print(args)

if __name__ == '__main__':
    main()
