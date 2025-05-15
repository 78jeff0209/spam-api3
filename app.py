
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
# 載入模型與向量器\ nmodel = joblib.load('spam_detector_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json or {}
    text = data.get('text', '')
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return jsonify({'label': 'spam' if pred==1 else 'ham'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
