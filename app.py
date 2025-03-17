import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from datetime import datetime, timedelta

# Set Streamlit page config
st.set_page_config(page_title="Data Science Tutor AI", layout="wide")

# Custom CSS for UI styling
st.markdown(
    """
    <style>
        body {background-color: #121212; color: white;}
        .stTextInput>div>div>input {color: white; background-color: #333; border-radius: 10px;}
        .stButton>button {background-color: #0057b7; color: white; border-radius: 10px;}
        .stChatMessage {border-radius: 12px; padding: 10px; margin: 5px;}
        .title {text-align: center; font-size: 24px; font-weight: bold;}
        .subtitle {text-align: center; font-size: 16px; margin-bottom: 20px; color: #bbb;}
        .sidebar-title {font-size: 18px; font-weight: bold;}
        .sidebar-section {border-bottom: 1px solid #555; padding-bottom: 10px; margin-bottom: 10px;}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar - API Key Input & Features
st.sidebar.markdown("<p class='sidebar-title'>Settings</p>", unsafe_allow_html=True)
gemini_api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")
if not gemini_api_key:
    st.sidebar.warning("Please enter your API key to proceed.")
    st.stop()

chat_model = ChatGoogleGenerativeAI(api_key=gemini_api_key, model="gemini-1.5-pro")

# Sidebar - Features Section
st.sidebar.markdown("<p class='sidebar-title'>Features</p>", unsafe_allow_html=True)
st.sidebar.markdown("""
- Conversational memory
- Code examples
- Data visualization explanations
- Statistical concepts
- Machine learning guidance
""")

# Sidebar - Example Questions
st.sidebar.markdown("<p class='sidebar-title'>Example Questions</p>", unsafe_allow_html=True)

st.sidebar.button("Explain bias-variance tradeoff in machine learning.")
st.sidebar.button("What are precision, recall, and F1-score in classification?")
st.sidebar.button("Explain overfitting and underfitting with examples.")
st.sidebar.button("What is a confusion matrix, and how is it used?")
st.sidebar.button("How does a decision tree algorithm work?")
st.sidebar.button("Explain how k-means clustering works.")


# Chat Prompt Setup
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are an AI Data Science Tutor. Answer only data science-related queries."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{human_input}")
])

output_parser = StrOutputParser()

# Memory Initialization
if "memory_buffer" not in st.session_state:
    st.session_state.memory_buffer = {"history": []}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def prepare_input(inputs):
    return {"chat_history": st.session_state.memory_buffer["history"], "human_input": inputs["human_input"]}

runnable_prepare_input = RunnableLambda(prepare_input)

# Execution Chain
chain = (runnable_prepare_input | chat_template | chat_model | output_parser)

def chat_with_bot(query):
    response = chain.invoke({"human_input": query})
    return response

def save_to_memory(query, response):
    timestamp = datetime.now()
    st.session_state.memory_buffer["history"].append(HumanMessage(content=query, additional_kwargs={"timestamp": timestamp}))
    st.session_state.memory_buffer["history"].append(AIMessage(content=response, additional_kwargs={"timestamp": timestamp}))
    st.session_state.chat_history.append((query, response, timestamp))

# UI Header
st.markdown("<p class='title'>Data Science Tutor AI</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Your personal AI assistant for learning data science.</p>", unsafe_allow_html=True)

# Chat Input Section
user_input = st.text_input("Ask a data science question:", "")
if st.button("Send") and user_input:
    response = chat_with_bot(user_input)
    save_to_memory(user_input, response)

# Display Chat History
current_date = datetime.now()
yesterday = current_date - timedelta(days=1)
seven_days_ago = current_date - timedelta(days=7)

def display_messages(label, date_filter):
    if any(date_filter(timestamp.date()) for _, _, timestamp in st.session_state.chat_history):
        st.subheader(label)
        for user_msg, bot_msg, timestamp in st.session_state.chat_history:
            if date_filter(timestamp.date()):
                with st.chat_message("user"):
                    st.write(user_msg)
                with st.chat_message("assistant"):
                    st.write(bot_msg)

display_messages("Today", lambda d: d == current_date.date())
display_messages("Yesterday", lambda d: d == yesterday.date())
display_messages("Previous 7 Days", lambda d: seven_days_ago.date() <= d < yesterday.date())

# Sidebar Chat History Download
st.sidebar.markdown("<p class='sidebar-title'>Chat History Tools</p>", unsafe_allow_html=True)
if st.session_state.chat_history:
    history_text = "\n".join([f"You: {user_msg}\nBot: {bot_msg}\n" for user_msg, bot_msg, _ in st.session_state.chat_history])
    st.sidebar.download_button("Download Chat History", data=history_text, file_name="chat_history.txt", mime="text/plain")


# API KEY = "AIzaSyDS2owlcE48NTAUacffTg-f7Vd9o3xbzMo"