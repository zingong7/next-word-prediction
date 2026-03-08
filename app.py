import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Next Word Predictor",
    layout="centered"
)

st.title("LSTM Next Word Predictor")
st.write("Type a phrase and let the model predict the next word.")


@st.cache_resource
def load_resources():
    model = load_model("next_word_predict_model.h5")
    
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    
    return model, tokenizer

model, tokenizer = load_resources()

def predict_next_word(model, tokenizer, text, max_sequence_len):

    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences([token_list],maxlen=max_sequence_len - 1,padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted)
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word, predicted[0][predicted_index]
    return None, None

def generate_text(text, num_words):

    max_sequence_len = model.input_shape[1] + 1

    current_text = text

    for _ in range(num_words):
        next_word, _ = predict_next_word(model,tokenizer,current_text,max_sequence_len)
        if next_word is None:
            break
        current_text += " " + next_word

    return current_text


input_text = st.text_input(
    "Enter your text",
    placeholder="Example: To be or not to"
)

num_words = st.slider(
    "How many words should the model generate?",
    1,
    10,
    3
)


if st.button("✨ Generate Text"):

    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:

        with st.spinner("Model is thinking..."):

            result = generate_text(input_text, num_words)

        st.success("Prediction Complete!")

        st.subheader("Generated Text")
        st.write(result)

        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append(result)


if "history" in st.session_state:

    st.subheader("Prediction History")

    for i, item in enumerate(reversed(st.session_state.history)):
        st.write(f"{i+1}. {item}")
