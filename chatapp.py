import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from dash import Dash, html, dcc, Input, Output, State

# Tải dữ liệu và mô hình
model = load_model('chatbot_model.h5')
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Hàm tiền xử lý câu hỏi
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Hàm chuyển câu hỏi thành dạng bag of words
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

# Hàm dự đoán lớp của câu hỏi
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

# Hàm lấy phản hồi dựa trên intent
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Hàm phản hồi chính cho chatbot
def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res

# Khởi tạo ứng dụng Dash
app = Dash(__name__)

# Thiết lập giao diện của ứng dụng
app.layout = html.Div([
    html.H1('Hỗ trợ sinh viên Hutech'),
    dcc.Textarea(
        id='question-area',
        placeholder='Nhập câu hỏi của bạn...',
        style={'width': '60%', 'height': 100}
    ),
    html.Button(id='submit-btn', children='Gửi câu hỏi'),
    html.Div(id='response-area', style={'margin-top': '20px'})
])

# Callback xử lý khi người dùng gửi câu hỏi
@app.callback(
    Output('response-area', 'children'),
    Input('submit-btn', 'n_clicks'),
    State('question-area', 'value')
)
def create_response(n_clicks, question):
    if question:
        response = chatbot_response(question)
        return response
    return "Hãy nhập câu hỏi của bạn."

# Chạy ứng dụng
if __name__ == '__main__':
    app.run_server(debug=True)
