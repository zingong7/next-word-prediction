# Next Word Prediction with LSTM


A deep learning project that predicts the next word in a sequence of text using an LSTM model, trained on Shakespeare's *Hamlet*. Deployed as an interactive web app using Streamlit.

---

## Features

- Predicts the next word given any input phrase
- Generates multiple words iteratively (1–10 words)
- Tracks prediction history in the same session
- Trained on Shakespeare's *Hamlet* via NLTK Gutenberg
- ~80% categorical accuracy


## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/your-repo.git
cd Word-prediction-Deep-learning
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the app**
```bash
streamlit run app.py
```

---

## Usage

1. Enter a phrase in the text box, e.g. `"To be or not to"`
2. Use the slider to choose how many words to generate
3. Click **Generate Text** and view the result


## Model Architecture

- **Embedding Layer:** `Embedding(total_words, 100)`
- **LSTM Layers:** 150 units → 100 units
- **Dropout:** 0.2
- **Output:** Dense + Softmax
- **Loss:** Categorical Cross-Entropy

---

## Folder Structure
```
Word-prediction-Deep-learning/
│
├── app.py
├── hamlet.txt
├── next_word_predict_model.h5
├── tokenizer.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Requirements

- Python 3.10+
- TensorFlow / Keras
- NumPy
- Streamlit
- NLTK
```bash
pip install -r requirements.txt
```

---

## Future Work

- Add Top-k / Top-p sampling for more natural output
- Train on a larger dataset
- Deploy on Streamlit Cloud

---

## License

This project is licensed under the MIT License.
