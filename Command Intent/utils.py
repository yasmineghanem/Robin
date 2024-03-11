import nlpaug.augmenter.word as naw
import numpy as np
import random
import nltk

def clean(line):
    cleaned_line = ''
    for char in line:
        if char.isalpha():
            cleaned_line += char
        else:
            cleaned_line += ' '
    cleaned_line = ' '.join(cleaned_line.split())
    return cleaned_line

def augment_data(text):
    pass