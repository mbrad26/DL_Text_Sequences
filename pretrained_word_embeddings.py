import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


base_dir = 'C:\/Users\mbrad\Downloads\kaggle\/aclImdb'
train_dir = os.path.join(base_dir, 'train')

texts = []
labels = []

for label_type in ['pos', 'neg']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding='utf8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)


maxlen = 100
max_words = 10000
train_samples = 200
val_samples = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)

sequence = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found {}s unique tokens')

data = pad_sequences(sequence, maxlen=maxlen)
labels = np.asarray(labels)
print('Data tensor shape: ', data.shape)
print('Label tensor shape: ', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:train_samples]
y_train = labels[: train_samples]
x_val = data[train_samples: train_samples + val_samples]
y_val = labels[train_samples: train_samples + val_samples]


glove_dir = 'C:\/Users\mbrad\Downloads\kaggle\glove'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found {}s word vectors'.format(len(embeddings_index)))


embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector





