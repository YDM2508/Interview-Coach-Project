import streamlit as st
import requests
skills_data = []
def main():
    global skills_data  # Declare skills_data as a global variable
    skills_data = [] 
    st.title("Profile Creation")

    option = st.radio("Select an option:", ("Upload Resume", "LinkedIn Profile", "Upload Both"), key="main_profile_option_radio")

    if option == "Upload Resume":
        # File upload
        resume_file = st.file_uploader("Upload your CV 📄", type=['docx', 'pdf'], key="main_resume_file_uploader")

        if resume_file is not None:
            if st.button("Submit", key="main_submit_resume_button"):
                parsed_data = parse_resume(resume_file.read())

                if parsed_data is not None:
                    form_data = display_form(parsed_data, skills_data)
                    skills_data.extend(parsed_data.get('skills', []))  # Update skills_data

    elif option == "LinkedIn Profile":
        linkedin_url = st.text_input("Enter LinkedIn Profile URL:", key="main_linkedin_url_input")
        if st.button("Extract Data", key="main_extract_data_button"):
            extracted_data = extract_linkedin_data(linkedin_url)

            if extracted_data is not None:
                st.subheader("LinkedIn Profile Data")
                display_linkedin_data(extracted_data)

    elif option == "Upload Both":
        # File upload for resume
        resume_file = st.file_uploader("Upload your CV 📄", type=['docx', 'pdf'], key="main_resume_file_uploader_both")

        # Text input for LinkedIn profile URL
        linkedin_url = st.text_input("Enter LinkedIn Profile URL:", key="main_linkedin_url_input_both")

        if st.button("Submit", key="main_submit_both_button"):
            if resume_file is not None:
                parsed_resume_data = parse_resume(resume_file.read())
                if parsed_resume_data is not None:
                    form_data = display_form(parsed_resume_data, skills_data)
                    skills_data.extend(parsed_resume_data.get('skills', []))  # Update skills_data

            if linkedin_url:
                extracted_data = extract_linkedin_data(linkedin_url)
                if extracted_data is not None:
                    st.subheader("LinkedIn Profile Data")
                    display_linkedin_data(extracted_data)
    # st.session_state.skills = ', '.join(skills_data)       
    return skills_data
def parse_resume(resume_content):
    api_url = 'https://api.apilayer.com/resume_parser/upload'
    api_key = 'x6z8QQuT1STz8tGLG5vbA8RpYHFLFDUu'
    # api_key = 'fgxfBfuA8d3tKBJpwypKgSactb1VGDtY'

    headers = {
        'Content-Type': 'application/octet-stream',
        'apikey': api_key
    }

    response = requests.post(api_url, headers=headers, data=resume_content)

    if response.status_code == 200:
        parsed_data = response.json()
        return parsed_data
    else:
        st.error(f"Error parsing the resume. Status code: {response.status_code}")
        return None

def extract_linkedin_data(linkedin_url):
    # api_key = "sk_live_6608db000474e0ef7231763c_key_fhrwxwldhhg"
    api_key = "sk_live_6609168a0474e0ef7231763d_key_4uf4rghyu09"
    url = "https://api.scrapin.io/enrichment/profile"
    querystring = {"apikey": api_key, "linkedinUrl": linkedin_url}

    response = requests.get(url, params=querystring)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        st.error("Failed to fetch LinkedIn Profile data. Status code: {}".format(response.status_code))
        return None

def display_form(parsed_data, skills_data):
    st.subheader("Candidate Details:")
    form_data = {}

    for key, value in parsed_data.items():
        form_data[key] = st.text_input(key, value)
        if key == 'skills':
            skills_data.extend(value)  # Update skills_data with the parsed skills
            if "pys" not in st.session_state:
                st.session_state.pys = value
                
            if "pys" in st.session_state:
                st.session_state.pys = value
            # if "pys" not in st.session_state:
            #     st.session_state.pys = value

    if st.button("Update Resume"):
        st.write("Profile Updated Successfully!")
        # You can save the updated data or perform further actions here

    return form_data

def display_linkedin_data(linkedin_data):
    # Display LinkedIn Profile data in an editable form
    for key, value in linkedin_data.items():
        if key not in ["success", "credits_left", "rate_limit_left"]:
            if isinstance(value, dict):
                st.subheader(key)
                for sub_key, sub_value in value.items():
                    linkedin_data[key][sub_key] = st.text_input(sub_key, value=sub_value)
            else:
                linkedin_data[key] = st.text_input(key, value=value)

    if st.button("Update LinkedIn Profile Data"):
        st.write("Profile Updated Successfully!")
        # You can save the updated data or perform further actions here


import streamlit as st
from streamlit_lottie import st_lottie
from typing import Literal
from dataclasses import dataclass
import json
import base64
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
import google.generativeai as gen_ai
import nltk
from prompts.prompts import templates
from IPython.display import Audio
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import speech_recognition as sr
import pyttsx3
from langchain.chains import ConversationChain, RetrievalQA
from typing import Generator
import re
from audio_recorder_streamlit import audio_recorder
from speech_recognition_openai.openai_whisper import save_wav_file, transcribe
from langchain.prompts.prompt import PromptTemplate
from prompts.prompts import templates

# Initialize the recognizer 
r = sr.Recognizer() 
 

# Function to convert text to


with st.sidebar:
        st.markdown('''
        ## About ℹ️
        Automating the interview coaching process using AI and speech interaction 🤖💬
        ''')
        st.write('Made by Smart Solvers 💡')
        import base64

        # file_ = open("images\sidebarlogo.gif", "rb")
        # contents = file_.read()
        # data_url = base64.b64encode(contents).decode("utf-8")
        # file_.close()

        # st.markdown(
        #     f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" width="300" height="500">',
        #     unsafe_allow_html=True,
        # )
        
skills_data = main()
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

with st.expander("""Why did I encounter errors when I tried to talk to the AI Interviewer?"""):
    st.write("""
    This is because the app failed to record. Make sure that your microphone is connected and that you have given permission to the browser to access your microphone.""")

st.markdown("""\n""")
st.markdown("""\n""")

str1 = ""

# Traverse the skills_data list and concatenate its elements into a single string
for ele in skills_data:
    str1 += ele


st.markdown("""
<style>
.stTextArea {
    display: none;
}
</style>
""", unsafe_allow_html=True)
# print(st.session_state.pys)
# Use the value of 'skills' session state variable as the default value for the text area
try:
    jd = st.text_area("Please enter the job description here (If you don't have one, enter keywords, such as Machine Learning or Python instead):: ",value=st.session_state.pys,label_visibility="hidden")
except Exception as e :
    print("ignore")
# if len(skills_data) != 0:
#     jd = st.text_area("""Please enter the job description here (If you don't have one, enter keywords, such as Machine Learning or Python instead):: """,value=skills_data)
auto_play = st.checkbox("Let AI interviewer speak! (Please don't switch during the interview)")

@dataclass
class Message:
    '''dataclass for keeping track of the messages'''
    origin: Literal["human", "ai"]
    message: str

def initialize_session_state():
    '''Initialize session state variables'''
    gen_ai.configure(api_key="AIzaSyDhKgqZ-_rqPr0A421CIVRvJk8l0MbUBKc")
    model = gen_ai.GenerativeModel('gemini-pro')    
    if "retriever" not in st.session_state:
        st.session_state.retriever = jd
    if "chain_type_kwargs" not in st.session_state:
        Interview_Prompt = f"now ask me next a quection based on {jd} in one line and stick to the topic for entire interview."
        st.session_state.jd_chain_type_kwargs = {"prompt": Interview_Prompt}
    # interview history
    if 'jd_memory' not in st.session_state:
        st.session_state.jd_memory = ConversationBufferMemory()
    if "history" not in st.session_state:
        st.session_state.history = []
        # Initial prompt to the model
        initial_prompt = f"You are the Generative AI based interview coach. dont give any kind of * mark in responce. Developed in PVG by students of the Information Technology department, your primary goal is to conduct interviews. As the interviewer (model), your role is to ask one question at a time, in one or two lines, without repeating the asked questions. The next question should be based on the quality response by the candidate. If the candidate answers correctly or know the answer, ask a tougher question in that topic. If the candidate gives a wrong answer or don't know the answer, ask an easier question in that topic. Your focus is to stay on the topic, and you should strictly adhere to the role of an interviewer without providing compliments.If candidate ask the quection don't provide answers provide responce like I am your interviewer; you can't ask me questions. Let's focus on Interview. now ask me a quection based on {jd} in one line and stick to the topic for entire interview."
        # Start chat session with initial prompt
        initial_response = model.generate_content(initial_prompt)
        # Add initial response to history
        st.session_state.history.append(Message("ai", initial_response.text))
        # Speak initial response
        SpeakText(initial_response.text)
    # token count
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory()
    if "guideline" not in st.session_state:
        st.session_state.guideline = "You are the Generative AI based interview coach. Developed in PVG by students of the Information Technology department, your primary goal is to conduct interviews. As the interviewer (model), your role is to ask one question at a time, in one or two lines, without repeating the asked questions. The next question should be based on the quality response by the candidate. If the candidate answers correctly or know the answer, ask a tougher question in that topic. If the candidate gives a wrong answer or don't know the answer, ask an easier question in that topic. Your focus is to stay on the topic, and you should strictly adhere to the role of an interviewer without providing compliments.If candidate ask the quection don't provide answers provide responce like I am your interviewer; you can't ask me questions. Let's focus on Interview."  # Replace with actual guideline
    # chat session and memory
    if "chat_session" not in st.session_state:
        gen_ai.configure(api_key="AIzaSyDhKgqZ-_rqPr0A421CIVRvJk8l0MbUBKc")
        model = gen_ai.GenerativeModel('gemini-pro')
        st.session_state.chat_session = model.start_chat(history=[])
    if "feedback" not in st.session_state:
        gen_ai.configure(api_key="AIzaSyDhKgqZ-_rqPr0A421CIVRvJk8l0MbUBKc")
        model = gen_ai.GenerativeModel('gemini-pro')
        st.session_state.feedback_session = model.start_chat(history=[])
    

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
            except:
                st.session_state.history.append(Message("ai", "Sorry, I didn't get that."))
                return "Please try again."
        else:
            input = human_answer

        st.session_state.history.append(
            Message("human", input)
        )
        # Construct the prompt based on conversation history and previous response
        conversation_history = "\n".join([f"{msg.origin.capitalize()}: {msg.message}" for msg in st.session_state.history])
        prompt = f"Based on the conversation history:\n\n{conversation_history}\n\nAnd the candidate's previous response:\n\n{input}\n\nAsk the next relevant interview question related to the job description: {jd}. Remember to follow the guidelines: {st.session_state.guideline}"

        # Gemini-Pro answer and save to history
        gemini_response = st.session_state.chat_session.send_message(prompt)
        llm_answer = gemini_response.text
        
        # Use pyttsx3 to synthesize speech
        SpeakText(llm_answer)

        # save audio data to history
        st.session_state.history.append(
            Message("ai", llm_answer)
        )
        st.session_state.token_count += cb.total_tokens

try:
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
        # Construct the chat history
            conversation_history = "\n".join([f"{msg.origin.capitalize()}: {msg.message}" for msg in st.session_state.history])
            
            # Provide the chat history to the feedback session
            feedback_response = st.session_state.feedback_session.send_message(f"Based on the chat history:\n\n{conversation_history}\n\nI would like you to evaluate the candidate based on the following format:\n\nSummarization: summarize the conversation in a short paragraph.\n\nPros: Give positive feedback to the candidate.\n\nCons: Tell the candidate what he/she can improves on.\n\nScore: Give a score to the candidate out of 100.\n\nSample Answers: sample answers to each of the questions in the interview guideline.")
            evaluation = feedback_response.text
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
except Exception as e:
    print("ignore")
if __name__ == "__main__":
    # main()
    pass
