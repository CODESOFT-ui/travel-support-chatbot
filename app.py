from langchain.llms.openai import OpenAI
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool
from langchain.agents import tool
from langchain.agents import AgentType
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.utilities import OpenWeatherMapAPIWrapper
from langchain.agents import initialize_agent
import os
import streamlit as st

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

@st.cache_resource(show_spinner=False)
def LLM_init(openai_api_key):
    os.environ['OPENWEATHERMAP_API_KEY'] = os.environ.get("OS.ENVIRON['OPENWEATHERMAP_API_KEY']", "")
    os.environ['SERPER_API_KEY'] = os.environ.get("OS.ENVIRON['SERPER_API_KEY']", "")
    os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', "")
    weather = OpenWeatherMapAPIWrapper()
    search = GoogleSerperAPIWrapper()
    llm = OpenAI(temperature=0)

    tools =[
        Tool(
            name="current search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of the world"
        ),
        Tool(
            name = "weather",
            func = weather.run,
            description = "Return current weather data based on location"
        ),
        Tool(
            name = "getlength",
            func = get_word_length,
            description = "Return the length of a word"
        ),
    ]

    memory = ConversationBufferMemory(memory_key="chat_history")
    
    llm_chain = initialize_agent(
        tools,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        llm=llm, 
        memory=memory, 
        verbose=True,
    )
    
    return llm_chain



 #Set page title and configuration
st.set_page_config(page_title="🦜🔗 Travel App ")

# Main title
st.title('🦜🔗 Travel App')

# Title for the chat assistant
st.title("💬 Travel Assistant")

# Sidebar to input OpenAI key
openai_key = st.sidebar.text_input("Enter OpenAI Key", "")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi, I am your travel consultant. How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Perform OpenAI processing
    if openai_key:  # Check if OpenAI key is provided
        llm_chain = LLM_init(openai_api_key=openai_key)
        msg = llm_chain(prompt)

        st.session_state.messages.append({"role": "assistant", "content": msg["output"]})
        st.chat_message("assistant").write(msg["output"])
    else:
        st.warning("Please input your OpenAI key in the sidebar.")
