import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model()

with open("tokenzier.pickle",'rb') as handle:
    tokenizer = pickle.load(handle)

    
def predict_next_word(seed_text, model, tokenizer, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    if len(token_list) >= max_sequence_len:
      token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted_probs,axis=1)
    for word ,index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None


st.title("Next Word Prediction with LSTM And Early Stopping")
input_text = st.text_input("Enter the sequence of words")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape(1)+1
    next_word=predict_next_word(input_text,model,tokenizer,max_sequence_len)
    st.write(next_word)