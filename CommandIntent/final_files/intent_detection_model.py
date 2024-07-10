import utils
import keras
import numpy as np
import tensorflow as tf


class IntentDetection(keras.Model):
    def __init__(self, num_classes, max_seq_len, vocab_size, embedding_dim=128, lstm_units=64, dropout_rate=0.4):
        super(IntentDetection, self).__init__()
        self.embedding = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embedding_dim),
        self.lstm = keras.layers.Bidirectional(
            keras.layers.LSTM(lstm_units, dropout=0.2)),
        self.dense_1 = keras.layers.Dense(lstm_units, activation='relu'),
        self.dropout = keras.layers.Dropout(dropout_rate),
        self.dense_2 = keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm(x)
        x = self.dense_1(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        return x


def train(model, train_data, val_data, epochs, path):
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'f1_score', 'precision', 'recall'])
    model.fit(train_data, validation_data=val_data, epochs=epochs)
    save_model(model, path)
    return model


def evaluate(model, test_data):
    return model.evaluate(test_data)

def save_model(model, path):
    model.save(path)

def load_model(path):
    return keras.models.load_model(path, custom_objects={'IntentDetection': IntentDetection})

def prepare_data(data):
    return data

def predict_intent(model, input):
    prediction = model.predict(input)
    return prediction
