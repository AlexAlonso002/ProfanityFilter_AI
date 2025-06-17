import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


df = pd.read_csv("Python_Models/word99.csv")

# Replace NaN values with empty strings
df["Words"] = df["Words"].fillna("").astype(str)
df["Label"] = df["Label"].fillna("").astype(str)

y_data = df["Label"]
x_data = df["Words"]


words = []
labels = []
for i in range(len(x_data)):
   words.append(x_data[i])
   labels.append(y_data[i])

import math
train = math.ceil(len(words) * 0.85)
print(train)

from sklearn.utils import shuffle
words, labels = shuffle(words, labels, random_state=44)

x_train = words[0:train]
y_train = labels[0:train]

x_test = words[train:]
y_test = labels[train:]
# Check if any non-string values exist
for i, x in enumerate(x_train):
    if not isinstance(x, str):
        print(f"Non-string value found at index {i}: {x} (type: {type(x)})")

tokenizer = Tokenizer(num_words=60, oov_token='<UNK>',char_level=True)
tokenizer.fit_on_texts(x_train)

# max len means the array will be 15 long and if the word is to short padded it with 0 after the word ends to keep array the same len
x_seq = tokenizer.texts_to_sequences(x_train)
x_pad = pad_sequences(x_seq, maxlen=20, padding='post', truncating='post')

x_seq_test = tokenizer.texts_to_sequences(x_test)
x_pad_test = pad_sequences(x_seq_test, maxlen=20, padding='post', truncating='post')

training_padded = np.array(x_pad)
training_labels = np.array(y_train)
testing_padded = np.array(x_pad_test)
testing_labels = np.array(y_test)

training_labels = np.array(y_train, dtype=np.int32)
testing_labels = np.array(y_test, dtype=np.int32)

#model cannot be exported to onnx 
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(60, 128),  # Higher vocab size for characters
    tf.keras.layers.Conv1D(64, 3, activation='relu'),  # CNN to capture character patterns
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True, activation='tanh')),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, activation='tanh')),
    tf.keras.layers.Dropout(0.3),  # Prevent overfitting
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])
# model to export to onnx
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(60, 128),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, activation='tanh', recurrent_initializer='glorot_uniform', unroll=True)
    ),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, activation='tanh', recurrent_initializer='glorot_uniform', unroll=True)
    ),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
Train_Model = model.fit(training_padded, training_labels, epochs=10)

loss, accuracy = model.evaluate(testing_padded, testing_labels, verbose=2)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")



#model.export("Profanity_model")

#python3 -m tf2onnx.convert --saved-model Profanity_model --output ProfanityOnnx.onnx --opset 13


#python -m tf2onnx.convert --saved-model Profanity_model --output ProfanityOnnx.onnx --opset 13


sentence = ["nigger","N1gger","Ni33er","Nigg3r","Nigg##","pussy","pu$$y","Faggot","sticky,","1","2","400","242","513","103913353","1024","alex's","Joseph's","it's","happ'ing","want","WANTED"]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=20, padding='post', truncating='post')

# Predict using the model
predictions = model.predict(padded)

# Loop through the predictions and print 0 for good, 1 for bad along with the prediction odds
for i, prediction in enumerate(predictions):
    print(f"Sentence: {sentence[i]}")
    # Print the prediction probability
    print(f"Prediction Probability: {prediction[0]:.4f}")
    # If the prediction probability is > 0.9, classify it as 1 (bad), otherwise 0 (good)
    if prediction[0] > 0.9:
        print("Prediction: 1 (Bad)")
    else:
        print("Prediction: 0 (Good)")

   
