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
@app.route('/analyze-all', methods=['POST'])
def analyze_all():
    try:
        # 收到圖片和文字
        image_file = request.files.get('image', None)
        text_input = request.form.get('text', '').strip()

        # 初始化文字容器
        extracted_text = ''

        # 有圖片就 OCR
        if image_file:
            ocr_response = requests.post(
                'https://api.ocr.space/parse/image',
                files={'filename': image_file},
                data={
                    'apikey': OCR_API_KEY,
                    'language': 'cht'
                }
            )
            result = ocr_response.json()
            if not result.get('IsErroredOnProcessing'):
                extracted_text = result['ParsedResults'][0].get('ParsedText', '')
            else:
                return jsonify({'error': 'OCR API 處理錯誤'}), 500

        # 合併文字（圖片 + 手動輸入）
        full_text = f"{extracted_text.strip()} {text_input}".strip()
        if not full_text:
            return jsonify({'error': '未提供有效文字'}), 400

        # 模型預測
        vec = vectorizer.transform([full_text])
        pred = model.predict(vec)[0]
        score = model.predict_proba(vec)[0][1]  # spam 的機率

        return jsonify({
            'final_label': 'spam' if pred == 1 else 'ham',
            'text': full_text,
            'total_score': round(score, 4)
        })

    except Exception as e:
        print(f"❌ 分析時錯誤：{e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
