import streamlit as st
from streamlit_lottie import st_lottie
from typing import Literal
from dataclasses import dataclass
import json
import base64
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain, RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import NLTKTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import nltk
from prompts.prompts import templates
from IPython.display import Audio
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import speech_recognition as sr
import pyttsx3
from typing import Generator
import re

# Initialize the recognizer 
r = sr.Recognizer() 
 
# Function to convert text to
# speech
def SpeakText(command):
     
    # Initialize the engine
    engine = pyttsx3.init(driverName='sapi5')
    voices = engine.getProperty('voices')

    # Print available voices and their properties (optional)
    for voice in voices:
        print(f"Voice: {voice.name}, ID: {voice.id}, Languages: {voice.languages}")

    # Set the desired voice (in this example, the second voice is used)
    engine.setProperty('voice', voices[1].id)
      # Set the speech rate (adjust the value to change the speed)
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 50)  # Decrease the rate by 50 to speak slower
    # engine.setProperty('rate', rate + 50)  # Increase the rate by 50 to speak faster
    engine.say(command) 
    engine.runAndWait()

def load_lottiefile(filepath: str):
    '''Load lottie animation file'''
    with open(filepath, "r") as f:
        return json.load(f)

st_lottie(load_lottiefile("images/welcome.json"), speed=1, reverse=False, loop=True, quality="high", height=300)

# st.markdown("""solutions to potential errors:""")
with st.expander("""Why did I encounter errors when I tried to talk to the AI Interviewer?"""):
    st.write("""
    This is because the app failed to record. Make sure that your microphone is connected and that you have given permission to the browser to access your microphone.""")

st.markdown("""\n""")
jd = st.text_area("""Please enter the job description here (If you don't have one, enter keywords, such as "communication" or "teamwork" instead): """)
auto_play = st.checkbox("Let AI interviewer speak! (Please don't switch during the interview)")

@dataclass
class Message:
    '''dataclass for keeping track of the messages'''
    origin: Literal["human", "ai"]
    message: str

def initialize_session_state():
    '''Initialize session state variables'''
    if "retriever" not in st.session_state:
        st.session_state.retriever = jd
    if "chain_type_kwargs" not in st.session_state:
        Behavioral_Prompt = PromptTemplate(input_variables=["context", "question"],
                                          template=templates.behavioral_template)
        st.session_state.chain_type_kwargs = {"prompt": Behavioral_Prompt}
    # interview history
    if "history" not in st.session_state:
        st.session_state.history = []
        st.session_state.history.append(Message("ai", "Hello there! I am your interviewer today. I will access your soft skills through a series of questions. Let's get started! Please start by saying hello or introducing yourself."))
        starttext = "Hello there! I am your interviewer today. I will access your soft skills through a series of questions. Let's get started! Please start by saying hello or introducing yourself."
        SpeakText(starttext)
        # engine = pyttsx3.init("sapi5")
        # engine.say("Hello there! I am your interviewer today. I will access your soft skills through a series of questions. Let's get started! Please start by saying hello or introducing yourself.")
        # engine.runAndWait()
    # token count
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory()
    if "guideline" not in st.session_state:
        llm = Ollama(
        model="gemma:2b",
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
    # llm chain and memory
    if "conversation" not in st.session_state:
        llm = Ollama(
        model="gemma:2b",
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
        PROMPT = PromptTemplate(
            input_variables=["history", "input"],
            template="""I want you to act as an interviewer strictly following the guideline in the current conversation.
                            Candidate has no idea what the guideline is.
                            Ask me questions and wait for my answers. Do not write explanations.
                            Ask question like a real person, only one question at a time.
                            Do not ask the same question.
                            Do not repeat the question.
                            Do ask follow-up questions if necessary. 
                            You name is GPTInterviewer.
                            I want you to only reply as an interviewer.
                            Do not write all the conversation at once.
                            If there is an error, point it out.

                            Current Conversation:
                            {history}

                            Candidate: {input}
                            AI: """)
        st.session_state.conversation = ConversationChain(prompt=PROMPT, llm=llm,
                                                       memory=st.session_state.memory)
    if "feedback" not in st.session_state:
        llm = Ollama(
        model="gemma:2b",
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
        st.session_state.feedback = ConversationChain(
            prompt=PromptTemplate(input_variables = ["history", "input"], template = templates.feedback_template),
            llm=llm,
            memory = st.session_state.memory,
        )

def answer_call_back():
    '''callback function for answering user input'''
    with get_openai_callback() as cb:
        # user input
        human_answer = st.session_state.answer
        # transcribe audio
        if voice:
            save_wav_file("temp/audio.wav", human_answer)
            try:
                input = transcribe("temp/audio.wav")
                # save human_answer to history
            except:
                st.session_state.history.append(Message("ai", "Sorry, I didn't get that."))
                return "Please try again."
        else:
            input = human_answer

        st.session_state.history.append(
            Message("human", input)
        )
        # OpenAI answer and save to history
        llm_answer = st.session_state.conversation.run(input)
        
        # Use pyttsx3 to synthesize speech
        SpeakText(llm_answer)

        # create audio widget with autoplay
        # audio_widget = Audio('temp/audio.mp3', autoplay=True)
        # save audio data to history
        st.session_state.history.append(
            Message("ai", llm_answer)
        )
        st.session_state.token_count += cb.total_tokens
        # return audio_widget

if jd:
    initialize_session_state()
    credit_card_placeholder = st.empty()
    col1, col2 = st.columns(2)
    with col1:
        feedback = st.button("Get Interview Feedback")
    with col2:
        guideline = st.button("Show me interview guideline!")
    audio = None
    chat_placeholder = st.container()
    answer_placeholder = st.container()

    if guideline:
        st.write(st.session_state.guideline)
    if feedback:
        evaluation = st.session_state.feedback.run("please give evalution regarding the interview")
        st.markdown(evaluation)
        st.download_button(label="Download Interview Feedback", data=evaluation, file_name="interview_feedback.txt")
        st.stop()
    else:
        with answer_placeholder:
            voice: bool = st.checkbox("I would like to speak with AI Interviewer!")
            if voice:
                answer = audio_recorder(pause_threshold=2.5, sample_rate=44100)
            else:
                answer = st.chat_input("Your answer")
            if answer:
                st.session_state['answer'] = answer
                audio = answer_call_back()
        with chat_placeholder:
            for answer in st.session_state.history:
                if answer.origin == 'ai':
                    if auto_play and audio:
                        with st.chat_message("assistant"):
                            st.write(answer.message)
                            st.write(audio)
                    else:
                        with st.chat_message("assistant"):
                            st.write(answer.message)
                else:
                    with st.chat_message("user"):
                        st.write(answer.message)

        credit_card_placeholder.caption(f"""
                        Progress: {int(len(st.session_state.history) / 30 * 100)}% completed.
        """)

else:
    st.info("Please submit job description to start interview.")
