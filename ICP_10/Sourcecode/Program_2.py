from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.layers import Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
import numpy as np
import matplotlib.pyplot as plt

#Changed the csv file where 'id' column name is missing
df = pd.read_csv('./imdb_master.csv', encoding='latin-1')

#print first 5 rows of the csv file
print(df.head())

#the values of the data frame 'review'
sentences = df['review'].values

#the values of the data frame label
y = df['label'].values

#printing the unique values from the label column
print(np.unique(y))

# tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)

# getting the vocabulary of data
max_review_len = max([len(s.split()) for s in sentences])

#defining the vocab size
vocab_size = len(tokenizer.word_index) + 1

sentences = tokenizer.texts_to_sequences(sentences)

#pads sentences in equal length
padded_docs = pad_sequences(sentences, maxlen=max_review_len)

#converting categorical data into numeric data
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

#Splitting the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(padded_docs, y, test_size=0.25, random_state=1000)

print(len(X_train))
# Number of features
print(vocab_size)

#Adding embedding layer in Keras
model = Sequential()

#We need to add flatten layer after the embedding layer
model.add(Embedding(vocab_size, 50, input_length=max_review_len))

#We need to add flatten layer after the embedding layer
model.add(Flatten())
model.add(layers.Dense(3, activation='softmax'))

#compiling the model and finding the accuracy
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=5, verbose=True, validation_data=(X_test, y_test), batch_size=256)

# Plot the graph for accuracy of both train and test data.
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Plot the graph for loss of both train and test data.
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()