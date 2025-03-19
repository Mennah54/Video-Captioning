import numpy as np
import os
import pickle
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, GlobalAveragePooling3D, Concatenate
from tensorflow.keras.utils import Sequence  


class VideoCaptionGenerator(Sequence):
    def __init__(self, features, captions, tokenizer, batch_size=32, max_len=20):
        self.features = features  # Dictionary: video_id -> feature vector
        self.captions = captions  # Dictionary: video_id -> list of captions
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.video_ids = list(self.features.keys())

    def __len__(self):
        return len(self.video_ids) // self.batch_size

    def __getitem__(self, idx):
        batch_ids = self.video_ids[idx * self.batch_size: (idx + 1) * self.batch_size]
        X1, X2, y = [], [], []

        for vid_id in batch_ids:
            feature = self.features[vid_id]
            captions = self.captions[vid_id]
            for cap in captions:
                seq = self.tokenizer.texts_to_sequences([cap])[0]
                for i in range(1, len(seq)):
                    in_seq = seq[:i]
                    out_seq = seq[i]

                    in_seq = np.pad(in_seq, (0, self.max_len - len(in_seq)), mode='constant')
                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)

        return [np.array(X1), np.array(X2)], np.array(y)

def build_model(vocab_size, max_len, feature_dim):
    # Input 1: Video Features
    inputs1 = Input(shape=(feature_dim,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Input 2: Text Sequence
    inputs2 = Input(shape=(max_len,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Merge
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model

#  3. Load Tokenizer, Max Length, Vocab Size
#with open(r"C:\Mennah\semster 8\deep learning\assigments\video captioning\main_project\tokenizer\max_length.txt", 'r') as f:
 #   max_length = int(f.read().strip())

#with open(r"C:\Mennah\semster 8\deep learning\assigments\video captioning\main_project\vocab_size.txt", 'r') as f:
 #   vocab_size = int(f.read().strip())

with open(r"C:\Mennah\semster 8\deep learning\assigments\video captioning\main_project\tokenizer\tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

data = np.load(r"C:\Mennah\semster 8\deep learning\assigments\video captioning\main_project\tokenizer\video_caption_pairs.npy", allow_pickle=True)

print(" Data shape/type:", type(data), data.shape)

vocab_size = len(tokenizer.word_index) + 1
max_len = 20
feature_dim = list(features.values())[0].shape[0]



# Generator
train_gen = VideoCaptionGenerator(features, captions, tokenizer, batch_size=32, max_len=max_len)

# Build and train model
model = build_model(vocab_size, max_len, feature_dim)
model.summary()

# Compile
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train + Print Loss and Accuracy
history = model.fit(train_gen, epochs=10, verbose=1)

# Print after training
for i in range(len(history.history['loss'])):
    print(f"Epoch {i+1}: Loss = {history.history['loss'][i]:.4f}, Accuracy = {history.history['accuracy'][i]:.4f}")