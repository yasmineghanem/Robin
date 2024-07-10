
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import json
import utils
import pandas as pd
import torch

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

class IntentData:
    '''
        This class is responsible for preparing the data for the intent detection model.
        It contains all the necessary functions to prepare the data for training and prediction.

        Args:
            - dataset_path (str) : the path to the dataset
            - train (bool) : whether to prepare the training data or not

        Attributes:
            - dataset_path (str) : the path to the dataset
            - tokenizer: the tokenizer to tokenize the sentences
            - intents (List[str]) : the unique intents in the dataset
            - num_of_intents (int) : the number of unique intents in the dataset (output classes)
            - intent_to_index (Dict[str, int]) : a dictionary mapping the intents to their corresponding index
            - index_to_intent (Dict[int, str]) : a dictionary mapping the index to the corresponding intent
            - vocab_size (int) : the size of the vocabulary
            - corpus (List[str]) : the sentences in the dataset
            - corpus_intents (List[str]) : the intents corresponding to each sentence in the dataset (class labels)
            - training_data (List[Tuple[np.array, np.array]]) : the training data in the form of a list of tuples
    '''

    def __init__(self, dataset_path=None, train=False):
        '''
            The constructor initializes the class members and sets the parameters of the class.

        '''
        self.dataset_path = dataset_path
        self.tokenizer = Tokenizer(filters='', oov_token='<unk>')
        self.__set_parameters()

        if train:
            self.prepare_training_data()

        # class members
        # datapath
        # intents
        # num_of_intents
        # intent_to_index
        # index_to_intent
        # vocab_size
        # corpus
        # corpus_intents
        # tokenizer
        # training_data

    def __set_parameters(self):
        with open(self.dataset_path, 'r') as f:
            dataset = json.load(f)

            # get all the unique intents from the dataset which are the keys of the dictionary
            self.intents = list(dataset.keys())
            self.num_of_intents = len(self.intents)

            # all the sentences in the dataset
            self.corpus = []

            # the intents corresponding to each sentence in the dataset
            self.corpus_intents = []

            for intent in self.intents:
                for keyword in dataset[intent]:
                    self.corpus.append(utils.clean(keyword))
                    self.corpus_intents.append(intent.lower())

        self.intent_to_index = {intent: index for index,
                                intent in enumerate(self.intents)}
        self.index_to_intent = {index: intent for index,
                                intent in enumerate(self.intents)}

        # fit the tokenizer on the corpus
        # this will create a vocabulary of all the words in the corpus
        self.tokenizer.fit_on_texts(self.corpus)

        # get the size of the vocabulary
        self.vocab_size = len(self.tokenizer.word_index) + 1

    def prepare_training_data(self):
        '''
            This function prepares the training data for the intent detection model.
        '''

        # maps the words in each sentence to their corresponding index
        sequences = self.tokenizer.texts_to_sequences(self.corpus)

        # final sentences form for training data
        # pads the sequences to the same length for training
        padded_sequences = keras.preprocessing.sequence.pad_sequences(
            sequences, padding='pre')

        # maps the target intents of each sentence to the corresponding index
        mapped_target_intents = [self.intent_to_index[intent]
                                 for intent in self.corpus_intents]

        # one hot encode the target intents
        # final target form for training data
        target_intents = keras.utils.to_categorical(
            mapped_target_intents, num_classes=self.num_of_intents)

        # create the training data
        # in the form of a list of tuples
        # each tuple contains the padded sequence and the target intent
        # List[Tuple[np.array, np.array]] => the first array is the padded sequence and the second array is the target intent
        # the shape of the padded sequence is (max_seq_len,) and the shape of the target intent is (num_of_intents,)
        self.training_data = list(zip(padded_sequences, target_intents))

    def split_data(self):
        '''
            This function splits the training data into training and validation sets.
        '''
        # split the data into training and validation sets
        train_data, val_data = train_test_split(
            self.training_data, test_size=0.2, random_state=42, shuffle=True)

        # convert the training and validation data into numpy arrays
        train_data = np.array(train_data)
        val_data = np.array(val_data)

        return train_data, val_data

    def prepare_input_for_prediction(self, input):
        '''
            This function takes a single sentence from the user and prepares it for prediction.
        '''
        # first we convert the text to sequences
        input = self.tokenizer.texts_to_sequences([input])

        # pad the sequences
        input = keras.preprocessing.sequence.pad_sequences(
            input, padding='pre')
        return input

    def get_intent(self, prediction):
        intent_index = np.argmax(prediction)
        target_intent = self.index_to_intent[intent_index]

        return target_intent


class NERData:
    '''
        This class is responsible for preparing the data for the named entity recognition model.
        It contains all the necessary functions to prepare the data for training and prediction.

        Args:
            - dataset_path (str) : the path to the dataset
            - intent_tag_path (str) : the path to the intent to tags mapping
            - train (bool) : whether to prepare the training data or not

        Attributes:
            - dataset_path (str) : the path to the dataset
            - intent_tags_path (str) : the path to the intent to tags mapping
            - dataset (List[Tuple[str, List[str], str]]) : the dataset in the form of a list of tuples
            - word_to_index (Dict[str, int]) : a dictionary mapping the words to their corresponding index
            - tag_to_index (Dict[str, int]) : a dictionary mapping the tags to their corresponding index
            - intent_to_index (Dict[str, int]) : a dictionary mapping the intents to their corresponding index
            - index_to_word (Dict[int, str]) : a dictionary mapping the index to the corresponding word
            - index_to_tag (Dict[int, str]) : a dictionary mapping the index to the corresponding tag
            - index_to_intent (Dict[int, str]) : a dictionary mapping the index to the corresponding intent
            - vocab_size (int) : the size of the vocabulary
            - num_of_tags (int) : the number of unique tags in the dataset (output classes)
            - tags (List[str]) : the unique tags in the dataset
            - num_of_intents (int) : the number of unique intents in the dataset
            - intent_to_tags (Dict[str, List[str]]) : a dictionary mapping the intents to their corresponding tags
    '''

    def __init__(self, dataset_path=None, intent_tag_path=None, train=False):
        self.dataset_path = dataset_path
        self.intent_tags_path = intent_tag_path
        self.__set_parameters()

        if train:
            self.prepare_training_data()
        
        # class members
        # dataset
        # word_to_index
        # tag_to_index
        # num_of_tags        

    def __set_parameters(self):
        self.read_data()
        self.word_to_index = {}
        self.tag_to_index = {}
        self.intent_to_index = {}

        # get all the unique words and tags from the dataset
        
        for sentence, tags, intent in self.dataset:
            for word in sentence.split():
                if word not in self.word_to_index:
                    self.word_to_index[word] = len(self.word_to_index)

            for tag in tags:
                if tag not in self.tag_to_index:
                    self.tag_to_index[tag] = len(self.tag_to_index)
            
            if intent not in self.intent_to_index:
                self.intent_to_index[intent] = len(self.intent_to_index)

        # add <UNK> for unkonwn words and tags
        self.word_to_index['<UNK>'] = len(self.word_to_index)
        self.tag_to_index['<UNK>'] = len(self.tag_to_index)

        # get all the tags 
        self.tags = list(self.tag_to_index.keys())

        # set the vocabulary size
        self.vocab_size = len(self.word_to_index)

        # the number of tags are the number of output classes
        self.num_of_tags = len(self.tag_to_index)

        # number of intents 
        self.num_of_intents = len(self.intent_to_index)

        # get the index-to dictionaries
        self.index_to_word = {index: word for word, index in self.word_to_index.items()}
        self.index_to_tag = {index: tag for tag, index in self.tag_to_index.items()}
        self.index_to_intent = {index: intent for intent, index in self.intent_to_index.items()}
        

    
    def read_data(self):
        data = pd.read_csv('../ner_dataset/ner_dataset.csv', encoding='latin1')

        # remove white spaces from column names
        # columns = ['Sentence #', 'Word', 'Tag', 'Intent']
        data.columns = data.columns.str.strip()

        # Group by 'Sentence #' and aggregate
        grouped_data = data.groupby('Sentence #').agg({
            'Word': lambda x: ''.join(x),  # Join words into a single sentence
            # Collect tags into a list
            'Tag': lambda x: list(x.str.strip()),
            # Collect intents into a list
            'Intent': lambda x: list(x.str.strip().str.replace('_', ' '))
        }).reset_index()  # Reset index to make 'Sentence #' a regular column

        self.dataset = []
        for _, row in grouped_data.iterrows():
            sentence = row['Word'][1:]
            tags = row['Tag']
            intents = row['Intent']
            self.dataset.append((sentence, tags, intents[0]))

        with open(self.intent_tags_path, 'r') as f:
            self.intent_to_tags = json.load(f)

    def check_data_validity(self):
        '''
            This function checks the validity of the data.
        '''
        for sentence, tags, _ in self.dataset:
            assert len(sentence.split()) == len(tags)
    
    def create_intent_mask(self, intent):
        intent_tags = self.intent_to_tags[intent]
        final_tags = []
        # create BI tags for the intent
        for tag in intent_tags:
            # print(tag)
            if tag == 'O': 
                final_tags.append(tag)
                continue
            
            final_tags.append('B-' + tag)
            final_tags.append('I-' + tag)

        mask = [tag in final_tags for tag in self.tags]
            
        mask = torch.tensor(mask, dtype=torch.long)

        return mask


    def prepare_training_data(self):
        # prepare the seequences of the sentences and tags
        # preapre tha masks for the intents

        training_data = []
        intent_masks = []

        for sentence, tags, intent in self.dataset:
            word_indices = [self.word_to_index.get(word, self.word_to_index['<UNK>']) for word in sentence.split()]
            word_indices_tensor = torch.tensor(word_indices, dtype=torch.long)
            tag_indices = [self.tag_to_index[tag] for tag in tags]
            tag_indices_tensor = torch.tensor(tag_indices, dtype=torch.long)
            intent_index = self.intent_to_index[intent]
            intent_index_tensor = torch.tensor(intent_index, dtype=torch.long)

            training_data.append((word_indices_tensor, tag_indices_tensor, intent_index_tensor))
            intent_masks.append(self.create_intent_mask(intent))


    def prepare_input_for_prediction(self, input, intent):
        word_indices = [self.word_to_index.get(word, self.word_to_index['<UNK>']) for word in input.split()]
        word_indices_tensor = torch.tensor(word_indices, dtype=torch.long)
        intent_index = self.intent_to_index[intent]
        intent_index_tensor = torch.tensor(intent_index, dtype=torch.long)
        
        intent_mask = self.create_intent_mask(intent)


        return word_indices_tensor, intent_index_tensor, intent_mask
    
    def get_entities(self, prediction):
        
        mapped_entities = [self.index_to_tag[index] for index in prediction]

        return mapped_entities 
