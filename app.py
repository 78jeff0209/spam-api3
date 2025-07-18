from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import requests
import os
from dotenv import load_dotenv

# 載入 .env 變數
load_dotenv()
OCR_API_KEY = os.environ.get("OCR_API_KEY")

app = Flask(__name__)
CORS(app)

# 載入模型與向量器
try:
    model = joblib.load('spam_detector_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except Exception as e:
    print(f"❌ 模型或向量器載入失敗：{e}")
    model = None
    vectorizer = None

# 文字預測 API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json or {}
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': '請提供 text 欄位'}), 400

        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        return jsonify({'label': 'spam' if pred == 1 else 'ham'})

    except Exception as e:
        print(f"❌ 預測錯誤：{e}")
        return jsonify({'error': str(e)}), 500

# 圖片 OCR + 預測 API（使用 OCR.space）
@app.route('/predict-image', methods=['POST'])
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': '請上傳圖片'}), 400

        image_file = request.files['image']

        ocr_response = requests.post(
            'https://api.ocr.space/parse/image',
            files={'filename': image_file},
            data={
                'apikey': OCR_API_KEY,
                'language': 'cht'
            }
        )

        result = ocr_response.json()
        if not result['IsErroredOnProcessing']:
            text = result['ParsedResults'][0]['ParsedText']
            if not text.strip():
                return jsonify({'error': '圖片無法擷取到有效文字'}), 400

            vec = vectorizer.transform([text])
            pred = model.predict(vec)[0]
            return jsonify({
                'label': 'spam' if pred == 1 else 'ham',
                'text': text.strip()
            })
        else:
            return jsonify({'error': 'OCR API 處理錯誤'}), 500
    except Exception as e:
        print(f"❌ 圖片處理錯誤：{e}")
        return jsonify({'error': str(e)}), 500
@app.route('/analyze-all', methods=['POST'])
def analyze_all():
    try:
        text_score = 0
        image_score = 0
        used_text = ''
        used_image_text = ''
        text_label = None
        image_label = None

        # 處理文字部分
        if 'text' in request.form and request.form['text'].strip():
            text = request.form['text'].strip()
            used_text = text
            vec = vectorizer.transform([text])
            proba = model.predict_proba(vec)[0]  # [ham, spam]
            spam_conf = proba[1]
            text_label = 'spam' if spam_conf > 0.5 else 'ham'

            if spam_conf > 0.75:
                text_score = 1.0
            elif spam_conf > 0.5:
                text_score = 0.5

        # 處理圖片部分
        if 'image' in request.files:
            image_file = request.files['image']
            ocr_response = requests.post(
                'https://api.ocr.space/parse/image',
                files={'filename': image_file},
                data={
                    'apikey': OCR_API_KEY,
                    'language': 'cht'
                }
            )

            result = ocr_response.json()
            if not result['IsErroredOnProcessing']:
                image_text = result['ParsedResults'][0]['ParsedText'].strip()
                used_image_text = image_text
                if image_text:
                    vec = vectorizer.transform([image_text])
                    proba = model.predict_proba(vec)[0]
                    spam_conf = proba[1]
                    image_label = 'spam' if spam_conf > 0.5 else 'ham'

                    if spam_conf > 0.75:
                        image_score = 1.5
                    elif spam_conf > 0.5:
                        image_score = 0.75
            else:
                image_label = 'error'

        total_score = text_score + image_score
        final_label = 'spam' if total_score >= 1.5 else 'ham'

        return jsonify({
            'final_label': final_label,
            'total_score': round(total_score, 2),
            'text': used_text,
            'text_score': text_score,
            'text_label': text_label,
            'image_text': used_image_text,
            'image_score': image_score,
            'image_label': image_label
        })

    except Exception as e:
        print(f"❌ analyze-all 錯誤：{e}")
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
