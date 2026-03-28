import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

# Khởi tạo các biến lưu trữ
words = []
classes = []
documents = []
ignore_letters = ['?', '!']

# Đọc dữ liệu từ tệp intents.json
data_file = open('intents.json', encoding='utf-8').read()
intents = json.loads(data_file)

# Tiền xử lý dữ liệu
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tách từ mỗi câu
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Thêm vào tài liệu
        documents.append((word_list, intent['tag']))
        # Thêm vào danh sách các lớp nếu chưa tồn tại
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, chuyển chữ thường và loại bỏ từ trùng lặp
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# In ra kết quả xử lý
print(f"{len(documents)} documents")
print(f"{len(classes)} classes: {classes}")
print(f"{len(words)} unique lemmatized words: {words}")

# Lưu dữ liệu đã xử lý vào tệp
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Tạo dữ liệu huấn luyện
training = []
output_empty = [0] * len(classes)

# Biến đổi các câu thành Bag of Words
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Xáo trộn dữ liệu huấn luyện và chuyển thành NumPy array
random.shuffle(training)
training = np.array(training, dtype=object)

# Tách dữ liệu thành đầu vào (X) và đầu ra (Y)
train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])

print("Training data created")

# Xây dựng mô hình với Keras
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Biên dịch mô hình
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Lưu mô hình đã huấn luyện
model.save('chatbot_model.h5')

print("Model created and saved")
