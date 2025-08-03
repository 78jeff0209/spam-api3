from transformers import BertTokenizer, BertForSequenceClassification
import torch

#載入 tokenizer 與模型
tokenizer = BertTokenizer.from_pretrained("ckiplab/bert-base-chinese")
model = BertForSequenceClassification.from_pretrained("ckiplab/bert-base-chinese", num_labels=2)

#假設你已微調完成並保存模型，可以改成 model = BertForSequenceClassification.from_pretrained("your_model_dir")
def predict_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred_label = torch.argmax(probs).item()
    return {
        "label": "spam" if pred_label == 1 else "ham",
        "score": float(probs[0][1])
    }