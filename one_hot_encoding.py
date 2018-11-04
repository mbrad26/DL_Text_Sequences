import numpy as np
import string
from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat', 'The dog ate my homework']

token_index = {}

for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1

print(token_index)

max_length = 10

results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))

print(max(token_index.values()))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1

print(results)

###########################################################################

characters = string.printable

token_index = dict(zip(range(1, len(characters) + 1), characters))

max_length = 50

results = np.zeros((len(samples), max_length, max(token_index.keys()) + 1))

for i, sample in enumerate(samples):
    for j, char in enumerate(sample):
        index = token_index.get(char)
        results[i, j, index] = 1


print(f'Characters {characters}')
print(f'Results {len(results)}')
print(f'Token Index {token_index}')

########################################################################

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)

sequences = tokenizer.texts_to_sequences(samples)
print('Sequences', sequences)

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
print('One hot results', one_hot_results)

word_index = tokenizer.word_index
print('Word index', word_index)

##########################################################################

dim = 1000
max_length = 10

results = np.zeros((len(samples), max_length, dim))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split())):
        index = abs(hash(word)) % dim
        results[i, j, index] = 1

print(f'Results: {results}')




















