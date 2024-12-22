import streamlit as st
from streamlit_option_menu import option_menu
from app_utils import switch_page
import streamlit as st
from PIL import Image

im = Image.open("images\sidebarlogo.gif")
st.set_page_config(page_title="AI Interviewer", layout="centered", page_icon=im)

home_title = "Generative AI-Based Interview Simulation And Analysis"
home_introduction = "Welcome to AI Interviewer, empowering your interview preparation with generative AI."
with st.sidebar:
    st.markdown('''
        ## About ℹ️
        Automating the interview coaching process using AI and speech interaction 🤖💬
        ''')
    st.write('Made by Smart Solvers 💡')
    import base64

st.markdown(f"""# {home_title} <span style=color:#2E9BF5><font size=5>Pro</font></span>""", unsafe_allow_html=True)
st.markdown("""\n""")
# st.markdown("#### Greetings")
st.markdown("Welcome to AI Interviewer! 👏 AI Interviewer is your personal interviewer powered by generative AI that conducts mock interviews."
            "You can upload your resume and enter job descriptions, and AI Interviewer will ask you customized questions. Additionally, you can configure your own Interviewer!")
st.markdown("""\n""")

st.markdown("""\n""")
st.markdown("#### Get started!")
st.markdown("Select one of the following screens to start your interview!")
selected = option_menu(
    menu_title=None,
    options=["Specific Skill Interview", "Technical Interview", "HR"],
    icons=["cast", "cloud-upload", "cast"],
    default_index=0,
    orientation="horizontal",
)
if selected == 'Specific Skill Interview':
    st.info("""
        📚In this session, the AI Interviewer will assess your Specific Skill Interview skills as they relate to the job description.
        
        - Each Interview will take 10 to 15 mins.
        - To start a new session, just refresh the page.
        - Choose your favorite interaction style (chat/voice)
        - Start introduce yourself and enjoy！ """)
    if st.button("Start Interview!"):
        switch_page("Specific Skill Interview")
if selected == 'Technical Interview':
    st.info("""
    📚In this session, the AI Interviewer will review your resume and discuss your past experiences.
    
    - Each Interview will take 10 to 15 mins.
    - To start a new session, just refresh the page.
    - Choose your favorite interaction style (chat/voice)
    - Start introduce yourself and enjoy！ """
    )
    if st.button("Start Interview!"):
        switch_page("Technical Interview")
if selected == 'HR':
    st.info("""
    📚In this session, the AI Interviewer will assess your soft skills as they relate to the job description.
    
    - Each Interview will take 10 to 15 mins.
    - To start a new session, just refresh the page.
    - Choose your favorite interaction style (chat/voice)
    - Start introduce yourself and enjoy！ 
    """)
    if st.button("Start Interview!"):
        switch_page("hr interview")

st.markdown("""\n""")
