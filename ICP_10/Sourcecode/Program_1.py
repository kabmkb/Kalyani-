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
print(vocab_size)

sentences = tokenizer.texts_to_sequences(sentences)

#pads sentences in equal length
padded_docs = pad_sequences(sentences, maxlen=max_review_len)

#converting categorical data into numeric data
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

#Splitting the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(padded_docs, y, test_size=0.25, random_state=1000)

# Number of features
# print(input_dim)
model = Sequential()

#Adding embedding layer in Keras
model.add(Embedding(vocab_size, 50, input_length=max_review_len))

#We need to add flatten layer after the embedding layer
model.add(Flatten())
model.add(layers.Dense(300, activation='relu'))

#no. of neurons in the last layer should be equal to the unique values in the label column
model.add(layers.Dense(3, activation='softmax'))

#compiling the model and finding the accuracy
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=2, verbose=True, validation_data=(X_test, y_test), batch_size=256)

