import torch
import torch.nn as nn
from TorchCRF import CRF


class BiLSTMCRF_Word(nn.Module):
    def __init__(self, vocab_size, word_embedding_dim, intent_embedding_dim, hidden_dim, output_dim, n_intents):

        super(BiLSTMCRF_Word, self).__init__()
        # hyperparameters
        self.word_embedding_dim = word_embedding_dim
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.output_dim = output_dim

        # model layers
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim)
        self.intent_embedding = nn.Embedding(n_intents, intent_embedding_dim)
        self.lstm = nn.LSTM(word_embedding_dim, hidden_dim //
                            2, bidirectional=True, dropout=0.2)
        self.hidden_to_tag = nn.Linear(hidden_dim, output_dim)
        self.crf = CRF(output_dim)

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2))

    def __get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        word_embeddings = self.word_embedding(
            sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(word_embeddings, self.hidden)
        lstm_features = self.hidden_to_tag(lstm_out)
        return lstm_features

    def neg_log_likelihood(self, sentence, tags):
        features = self.__get_lstm_features(sentence)
        print(features.shape)
        tags = tags.view(-1, 1)
        loss = -self.crf(features, tags)
        return loss

    def forward(self, sentence):
        lstm_features = self.__get_lstm_features(sentence)
        print(lstm_features.shape)
        tag_sequence = self.crf.decode(lstm_features)
        return tag_sequence
