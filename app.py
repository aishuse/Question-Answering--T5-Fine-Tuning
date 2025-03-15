import streamlit as st
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration

# Load tokenizer and model from the current directory
tokenizer = T5TokenizerFast.from_pretrained(".")
model = T5ForConditionalGeneration.from_pretrained(".")

st.set_page_config(page_title="T5 Question Answering", page_icon="🤖", layout="centered")

st.title("T5 Question Answering Model")

st.write("""
    This is a simple app that uses a trained T5 model to answer questions based on provided context.
    Please input your question and context below, and the model will provide an answer.
""")

context = st.text_area("Enter the context:")
question = st.text_input("Enter your question:")

def generate_answer(question, context):
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=150)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if st.button("Get Answer"):
    if context and question:
        answer = generate_answer(question, context)
        st.write(f"**Answer:** {answer}")
    else:
        st.warning("Please enter both a question and context.")
