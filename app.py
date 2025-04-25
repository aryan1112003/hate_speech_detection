import torch
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification
import gradio as gr

model_path = "./saved_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

df = pd.read_csv("single_word_abuse_dataset.csv")

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['label_enc'] = label_encoder.fit_transform(df['label'])

from sklearn.model_selection import train_test_split
_, val_texts, _, val_labels = train_test_split(
    df['text'], df['label_enc'], test_size=0.2, stratify=df['label_enc'], random_state=42
)

def evaluate_model():
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for text, label in zip(val_texts, val_labels):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=1).item()
            predictions.append(pred)
            true_labels.append(label)

    report = classification_report(true_labels, predictions, target_names=label_encoder.classes_)

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')

    metrics = f"""
    Accuracy: {accuracy:.4f}
    Precision: {precision:.4f}
    Recall: {recall:.4f}
    F1-Score: {f1:.4f}
    \n\nClassification Report:\n{report}
    """
    return metrics

def predict_hate_speech(text):
    model.eval()
    lines = text.strip().split('\n')
    hate_labels = ['abusive', 'hate', 'offensive']  # Adjust this list as per your dataset labels

    hate_count = 0
    predictions = []

    for line in lines:
        if not line.strip():
            continue  # skip empty lines
        inputs = tokenizer(line, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_label = torch.argmax(probs, dim=1).item()
        predicted_class = label_encoder.inverse_transform([pred_label])[0]
        predictions.append(f"Line: \"{line.strip()}\"\n→ Prediction: {predicted_class}")
        if predicted_class.lower() in hate_labels:
            hate_count += 1

    result = "\n\n".join(predictions)
    summary = f"\n\n🔍 Total Hate/Abusive Lines Detected: {hate_count} / {len(lines)}"
    return result + summary

def app_interface(text):
    prediction = predict_hate_speech(text)
    metrics = evaluate_model()
    return f"{prediction}\n\n📊 Model Evaluation:\n{metrics}"

iface = gr.Interface(
    fn=app_interface,
    inputs=gr.Textbox(lines=10, placeholder="Enter multiple lines of text, one per line..."),
    outputs="text",
    title="💬 Hate Speech Detection (BERT)",
    description="This app uses a fine-tuned BERT model to classify each line of input as hate/abusive or not.\n\nIt also displays classification metrics and counts how many hateful lines were found."
)

iface.launch()
