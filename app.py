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
model = None
vectorizer = None
try:
    model = joblib.load('spam_detector_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except Exception as e:
    print(f"❌ 模型或向量器載入失敗：{e}")
    # 如果載入失敗，model 和 vectorizer 會保持為 None，後續路由會檢查

# 文字預測 API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 檢查模型和向量器是否已載入
        if model is None or vectorizer is None:
            return jsonify({'error': '伺服器模型尚未載入，請稍後再試或聯繫管理員'}), 500

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
        # 檢查模型和向量器是否已載入
        if model is None or vectorizer is None:
            return jsonify({'error': '伺服器模型尚未載入，請稍後再試或聯繫管理員'}), 500

        if 'image' not in request.files:
            return jsonify({'error': '請上傳圖片檔案'}), 400

        image_file = request.files['image']

        # 檢查檔案是否為空（使用者選擇了檔案但內容為空）
        if image_file.filename == '':
            return jsonify({'error': '未選擇任何圖片檔案'}), 400

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
            # 打印 OCR API 的錯誤訊息以供除錯
            print(f"OCR API 處理錯誤：{result.get('ErrorMessage', '未知錯誤')}")
            return jsonify({'error': f"OCR API 處理錯誤: {result.get('ErrorMessage', '未知錯誤')}"}), 500
    except Exception as e:
        print(f"❌ 圖片處理錯誤：{e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-all', methods=['POST'])
def analyze_all():
    try:
        # 檢查模型和向量器是否已載入
        if model is None or vectorizer is None:
            return jsonify({'error': '伺服器模型尚未載入，請稍後再試或聯繫管理員'}), 500

        text_score = 0
        image_score = 0
        used_text = ''
        used_image_text = ''
        text_label = None
        image_label = None

        # 處理文字部分
        # 使用 .get() 方法更安全地獲取表單數據
        text_input_from_form = request.form.get('text', '').strip()
        if text_input_from_form:
            used_text = text_input_from_form
            vec = vectorizer.transform([used_text])
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
            # 檢查圖片檔案是否真的有內容
            if image_file.filename != '': # 確保檔案名不為空
                ocr_response = requests.post(
                    'https://api.ocr.space/parse/image',
                    files={'filename': image_file},
                    data={
                        'apikey': OCR_API_KEY,
                        'language': 'cht'
                    }
                )

                try:
                    result = ocr_response.json()
                except Exception as e:
                    print("❌ OCR API 回傳格式錯誤：", ocr_response.text)
                    return jsonify({'error': 'OCR_API_ERROR', 'details': ocr_response.text}), 500
                if not result['IsErroredOnProcessing']:
                    image_text = result['ParsedResults'][0]['ParsedText'].strip()
                    used_image_text = image_text
                    if image_text: # 檢查 OCR 是否提取到有效文字
                        vec = vectorizer.transform([image_text])
                        proba = model.predict_proba(vec)[0]
                        spam_conf = proba[1]
                        image_label = 'spam' if spam_conf > 0.5 else 'ham'

                        if spam_conf > 0.75:
                            image_score = 1.5
                        elif spam_conf > 0.5:
                            image_score = 0.75
                    else:
                        image_label = 'no_text_in_image' # OCR 處理成功但未找到文字
                else:
                    image_label = 'ocr_api_error' # OCR API 返回錯誤

        total_score = text_score + image_score
        if text_label and not image_label:
            final_label = text_label
        elif image_label and not text_label:
            final_label = image_label
        elif text_label or image_label:
            final_label = 'spam' if total_score >= 1.5 else 'ham'
        else:
            final_label = 'unknown'  # 兩者都沒有

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
