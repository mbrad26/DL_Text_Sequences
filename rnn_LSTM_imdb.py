import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

train = pad_sequences(x_train, maxlen=500)
test = pad_sequences(x_test, maxlen=500)

model = Sequential()
model.add(Embedding(10000, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

results = model.evaluate(test, y_test)
print(f'Results: {results}')

hist_dict = history.history

loss = hist_dict['loss']
acc = hist_dict['acc']
val_loss = hist_dict['val_loss']
val_acc = hist_dict['val_acc']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Loss')
plt.plot(epochs, val_loss, 'b', label='Val Loss')
plt.legend()
plt.figure()

plt.plot(epochs, acc, 'bo', label='Acc')
plt.plot(epochs, val_acc, 'b', label='Val Acc')
plt.legend()
plt.show()

