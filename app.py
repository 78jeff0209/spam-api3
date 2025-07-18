from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import requests
import os
from dotenv import load_dotenv

load_dotenv()
OCR_API_KEY = os.environ.get("OCR_API_KEY")

app = Flask(__name__)
CORS(app)

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

# 以下為 analyze_all 函式的佔位符，您需要根據實際邏輯來實現 predict_text_helper, predict_image_helper, 和 fuse_prediction_helper 函式。
# 通常不會直接從一個路由函式呼叫另一個路由函式來獲取結果。
# 建議將核心邏輯提取為輔助函式，然後在路由和 analyze_all 中呼叫這些輔助函式。

def predict_text_helper(text_input):
    # 請在此處實現文字預測的核心邏輯，例如：
    if not text_input:
        return {"label": "無文字", "confidence": 0.0}
    vec = vectorizer.transform([text_input])
    pred = model.predict(vec)[0]
    return {"label": "spam" if pred == 1 else "ham", "confidence": 0.0} # 請填寫實際的信心度

def predict_image_helper(image_file_input):
    # 請在此處實現圖片預測的核心邏輯，例如：
    ocr_response = requests.post(
        'https://api.ocr.space/parse/image',
        files={'filename': image_file_input},
        data={
            'apikey': OCR_API_KEY,
            'language': 'cht'
        }
    )
    result = ocr_response.json()
    if not result['IsErroredOnProcessing']:
        text = result['ParsedResults'][0]['ParsedText']
        if not text.strip():
            return {"label": "圖片無文字", "confidence": 0.0}
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        return {"label": "spam" if pred == 1 else "ham", "text": text.strip(), "confidence": 0.0} # 請填寫實際的信心度
    else:
        return {"label": "OCR錯誤", "confidence": 0.0, "error": result.get('ErrorMessage', '未知OCR錯誤')}

def fuse_prediction_helper(text_res, image_res):
    # 請在此處實現文字和圖片預測結果的融合邏輯
    # 這裡只是一個示例，您可以根據您的融合策略修改
    if text_res.get("label") == "spam" or image_res.get("label") == "spam":
        return {"label": "spam", "final_confidence": max(text_res.get("confidence", 0), image_res.get("confidence", 0))}
    return {"label": "ham", "final_confidence": 0.0}

@app.route("/analyze-all", methods=["POST"])
def analyze_all():
    try:
        text = request.form.get("text")
        image = request.files.get("image")

        text_result = {"label": "正常", "confidence": 0.0}
        image_result = {"label": "正常", "confidence": 0.0}

        if text:
            text_result = predict_text_helper(text)
        if image:
            image_result = predict_image_helper(image)

        final_result = fuse_prediction_helper(text_result, image_result)
        return jsonify({
            "text_result": text_result,
            "image_result": image_result,
            "final_result": final_result
        })
    except Exception as e:
        print(f"❌ analyze-all 處理錯誤：{e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)