from dataclasses import dataclass
import streamlit as st
from langchain.memory import ConversationBufferMemory
import google.generativeai as gen_ai
import pyttsx3
from typing import Literal
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import NLTKTextSplitter
import nltk
from vertexai.language_models import TextEmbeddingModel
import spacy

import numpy as np
import faiss
# Initialize Gemini-Pro
gen_ai.configure(api_key="AIzaSyDhKgqZ-_rqPr0A421CIVRvJk8l0MbUBKc")
model = gen_ai.GenerativeModel('gemini-pro')

@dataclass
class Message:
    """Class for keeping track of interview history."""
    origin: str
    message: str

def initialize_session_state_resume(resume_text):
    """Initialize session state variables for resume interview."""
    # Initialize conversation history with the introduction
    st.session_state.resume_history = [Message(origin="ai", message="Hello, I am your interviewer today. I will ask you some questions regarding your resume and your experience. Please start by saying hello or introducing yourself. Note: The maximum length of your answer is 4097 tokens!")]

    # Store resume text in session state
    st.session_state.resume_text = resume_text

def answer_call_back():
    """Callback function for answering user input."""
    # Get user input
    human_answer = st.session_state.answer

    # Add user input to conversation history
    st.session_state.resume_history.append(Message("human", human_answer))

    # Construct prompt for Gemini-Pro
    conversation_history = "\n".join([f"{msg.origin.capitalize()}: {msg.message}" for msg in st.session_state.resume_history])
    prompt = f"Based on the conversation history:\n\n{conversation_history}\n\nPlease ask the next relevant interview question."

    # Get response from Gemini-Pro
    gemini_response = model.generate_content(prompt)
    llm_answer = gemini_response.text

    # Add Gemini-Pro response to conversation history
    st.session_state.resume_history.append(Message("ai", llm_answer))

    return llm_answer

# UI components
st.title("Gemini-Pro Resume Interview")

with st.expander("Interview Guidelines"):
    st.write("These are the guidelines for conducting the interview.")

position = st.selectbox("Select the position you are applying for", ["Data Analyst", "Software Engineer", "Marketing"])
resume = st.file_uploader("Upload your resume", type=["pdf"])
auto_play = st.checkbox("Let AI interviewer speak! (Please don't switch during the interview)")

if position and resume:
    # Convert resume file to text
    resume_text = ""
    for page in PdfReader(resume).pages:
        resume_text += page.extract_text()

    # Initialize session state
    initialize_session_state_resume(resume_text)
    credit_card_placeholder = st.empty()
    col1, col2 = st.columns(2)
    with col1:
        feedback = st.button("Get Interview Feedback")
    with col2:
        guideline = st.button("Show me interview guideline!")
    chat_placeholder = st.container()
    answer_placeholder = st.container()
    audio = None

    if guideline:
        # Show interview guideline
        st.write("Create an interview guideline and prepare only two questions for each topic. Make sure the questions test the candidate's knowledge.")
    if feedback:
        # You can add feedback functionality here
        st.stop()
    else:
        with answer_placeholder:
            voice: bool = st.checkbox("I would like to speak with AI Interviewer!")
            if voice:
                answer = audio_recorder(pause_threshold=2, sample_rate=44100)
            else:
                answer = st.text_input("Your answer")
            if answer:
                st.session_state['answer'] = answer
                audio = answer_call_back()

        with chat_placeholder:
            for answer in st.session_state.resume_history:
                if answer.origin == 'ai':
                    if auto_play and audio:
                        st.write("Interviewer:", answer.message)
                        st.audio(audio, format='audio/wav')
                    else:
                        st.write("Interviewer:", answer.message)
                else:
                    st.write("Candidate:", answer.message)

        credit_card_placeholder.caption(f"Progress: {int(len(st.session_state.resume_history) / 30 * 100)}% completed.")