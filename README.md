🤖 Student Support Chatbot (Hutech)
👋 Giới thiệu

Student Support Chatbot là một hệ thống chatbot thông minh được xây dựng bằng Python và Machine Learning, giúp tự động hóa việc hỗ trợ sinh viên thông qua việc trả lời các câu hỏi thường gặp.

Hệ thống sử dụng mô hình học máy để phân loại ý định (intent classification) và đưa ra phản hồi phù hợp theo ngữ cảnh câu hỏi.

🎯 Mục tiêu dự án
Tự động hóa việc hỗ trợ sinh viên
Giảm tải cho bộ phận tư vấn
Cung cấp phản hồi nhanh chóng, chính xác
Ứng dụng Machine Learning vào bài toán thực tế
✨ Tính năng chính

🔹 💬 Chat thông minh

Nhận câu hỏi tự nhiên từ người dùng
Phân tích và hiểu ý định

🔹 🧠 Machine Learning

Sử dụng mô hình Neural Network (Keras)
Huấn luyện từ dữ liệu intents.json

🔹 📊 Xử lý ngôn ngữ tự nhiên (NLP)

Tokenization (tách từ)
Lemmatization (chuẩn hóa từ)
Bag of Words

🔹 🌐 Giao diện Web

Xây dựng bằng Dash (Plotly)
Giao diện đơn giản, dễ sử dụng
🧠 Cách hoạt động
Người dùng nhập câu hỏi
Hệ thống xử lý NLP:
Tách từ (NLTK)
Chuẩn hóa từ
Chuyển thành Bag of Words
Đưa vào mô hình Neural Network
Dự đoán intent
Trả về phản hồi phù hợp


🛠️ Công nghệ sử dụng
Công nghệ	Mô tả
🐍 Python	Ngôn ngữ chính
🧠 Keras / TensorFlow	Machine Learning
📚 NLTK	Xử lý ngôn ngữ tự nhiên
🌐 Dash	Web UI
📦 Pickle	Lưu dữ liệu
⚙️ Cài đặt & chạy dự án
1️⃣ Cài thư viện
pip install nltk tensorflow keras dash numpy

2️⃣ Tải dữ liệu NLTK
import nltk
nltk.download('punkt')
nltk.download('wordnet')

3️⃣ Huấn luyện mô hình
python train_model.py

4️⃣ Chạy chatbot
python app.py




🚀 Hướng phát triển
🔥 Tích hợp API (Flask / FastAPI)
💬 Kết nối với website thật
🧠 Nâng cấp lên Deep Learning / LLM
🌍 Hỗ trợ đa ngôn ngữ (Vietnamese NLP)
📱 Tích hợp mobile app
💡 Điểm nổi bật

✔ Tự xây dựng pipeline NLP từ đầu
✔ Hiểu rõ cách hoạt động của chatbot truyền thống
✔ Có thể mở rộng thành AI chatbot thực tế
✔ Phù hợp showcase khi apply Intern/Fresher AI/Backend

👤 Tác giả

Lê Quốc Bình
