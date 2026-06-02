# SENTIMENT ANALYSIS
### 4-Class NLP Classifier · 93.78% Train Accuracy · 87.53% Val Accuracy

---

> Classifies any text as **Positive**, **Negative**, **Neutral**, or **Irrelevant** — with confidence scores.  
> Built with TensorFlow/Keras, trained on 922 samples, 10 epochs.

---

## RESULTS

| Sample Input | Predicted Label | Confidence |
|---|---|---|
| "I love this game!" | ✅ Positive | 94.4% |
| "This is the worst update ever." | ❌ Negative | 98.8% |
| "It's an okay product." | ❌ Negative | 63.7% |
| "Bananas are yellow." | ⬜ Irrelevant | 96.6% |

---

## TRAINING METRICS

| Epoch | Train Accuracy | Val Accuracy | Train Loss | Val Loss |
|---|---|---|---|---|
| 1 | 60.52% | 77.08% | 0.9615 | 0.6183 |
| 3 | 88.92% | 86.10% | 0.3174 | 0.4029 |
| 5 | 92.39% | 87.34% | 0.2147 | 0.3859 |
| 7 | 93.10% | 87.53% | 0.1913 | 0.3922 |
| **10** | **93.78%** | **87.50%** | **0.1704** | **0.4153** |

---

## STACK

```
Python · TensorFlow · Keras · NumPy · Scikit-learn
Tokenizer · Pad Sequences · LabelEncoder · Embedding Layer
```

---

## ARCHITECTURE

```
Input Text
    ↓
Tokenizer + Pad Sequences (max_len)
    ↓
Embedding Layer
    ↓
LSTM / Dense Layers
    ↓
Softmax (4 classes)
    ↓
Label: Positive / Negative / Neutral / Irrelevant + Confidence %
```

---

## INSTALL

```bash
# Clone
git clone https://github.com/porasnehra/sentimental-analysis.git
cd sentimental-analysis

# Install dependencies
pip install tensorflow numpy scikit-learn
```

---

## USAGE

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Load model and tokenizer
model = load_model("sentiment_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

def predict_sentiment(text, max_len=100):
    seq = tokenizer.texts_to_sequences([text.lower()])
    padded = pad_sequences(seq, maxlen=max_len)
    prediction = model.predict(padded, verbose=0)
    class_idx = np.argmax(prediction)
    sentiment_label = le.inverse_transform([class_idx])[0]
    confidence = np.max(prediction)
    return f"{sentiment_label} ({confidence:.1%})"

# Example
samples = [
    "I love this game!",
    "This is the worst update ever.",
    "It's an okay product.",
    "Bananas are yellow."
]

for s in samples:
    print(f"'{s}' -> {predict_sentiment(s)}")
```

**Output:**
```
'I love this game!' -> Positive (94.4%)
'This is the worst update ever.' -> Negative (98.8%)
'It's an okay product.' -> Negative (63.7%)
'Bananas are yellow.' -> Irrelevant (96.6%)
```

---

## FEATURES

- **4-class classification** — goes beyond binary positive/negative
- **Confidence scores** — every prediction comes with a probability
- **Fast inference** — single-text predictions in milliseconds
- **Extensible** — swap the dataset and retrain for any domain

---

## API INTEGRATION (Optional)

```python
import requests

API_KEY = "YOUR_API_KEY"  # Replace with your key

headers = {"Authorization": f"Bearer {API_KEY}"}
payload = {"text": "I love this game!"}

response = requests.post(
    "https://your-api-endpoint.com/predict",
    json=payload,
    headers=headers
)

print(response.json())  # {"label": "Positive", "confidence": 0.944}
```

---

## AUTHOR

**Poras Nehra** · B.Tech CS @ MMEC, Mullana  
[GitHub](https://github.com/porasnehra) · [LinkedIn](https://linkedin.com/in/poras-nehra-142170367)

---

⭐ Star this repo if it helped you!
