from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# 嘗試載入模型與向量器
try:
    model = joblib.load('spam_detector_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except Exception as e:
    print(f"❌ 模型或向量器載入失敗：{e}")
    model = None
    vectorizer = None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model or not vectorizer:
            return jsonify({'error': '模型未正確載入'}), 500

        data = request.json or {}
        print("📩 收到資料：", data)
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': '請提供 text 欄位'}), 400

        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        return jsonify({'label': 'spam' if pred == 1 else 'ham'})

    except Exception as e:
        print(f"❌ 預測時錯誤：{e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

