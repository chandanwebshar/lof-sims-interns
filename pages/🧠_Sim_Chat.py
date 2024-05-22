import streamlit as st
from sim_prompts import *
import markdown2
from groq import Groq
from openai import OpenAI
import os

from audio_recorder_streamlit import audio_recorder
from prompts import *
import tempfile
import requests
import json
import base64
import random
from sims import llm_call, generate_pdf

st.set_page_config(page_title='Simulated Chat', layout = 'centered', page_icon = ':stethoscope:', initial_sidebar_state = 'expanded')

def assign_random_voice(sex):
    """
    Randomly assigns one of the specified strings to the variable 'voice'.

    Returns:
    - str: The assigned voice.
    
    The possible voices are 'alloy', 'echo', 'fable', 'onyx', 'nova', and 'shimmer'.
    """
    # List of possible voices
    male_voices = [ 'echo', 'fable', 'onyx' ]
    female_voices = ['alloy',  'nova', 'shimmer']
    
    if sex == 'male':
        voices = male_voices
    else:
        voices = female_voices
    
    # Randomly choose one voice from the list
    voice = random.choice(voices)
    
    return voice



def transcribe_audio(audio_file_path):
    from openai import OpenAI
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(    
        base_url="https://api.openai.com/v1",
        api_key=api_key,
    )
    audio_file = open(audio_file_path, "rb")
    transcript = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file, 
    response_format="text"
    )
    return transcript

def talk_stream(model, voice, input):
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(    
        base_url="https://api.openai.com/v1",
        api_key=api_key,
    )
    response = client.audio.speech.create(
    model= model,
    voice= voice,
    input= input,
    )
    response.stream_to_file("last_interviewer.mp3")
    
def autoplay_local_audio(filepath: str):
    # Read the audio file from the local file system
    with open(filepath, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    md = f"""
        <audio controls autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(
        md,
        unsafe_allow_html=True,
    )

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.write("*Please contact David Liebovitz, MD if you need an updated password for access.*")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True
def parse_groq_stream(stream):
    for chunk in stream:
        if chunk.choices:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

@st.cache_data
def extract_patient_door_chart_section(text):
    """
    Extracts the PATIENT DOOR CHART section from the given text string and returns it.
    
    Args:
    - text (str): The input text containing multiple sections, including "PATIENT DOOR CHART".
    
    Returns:
    - str: The extracted "PATIENT DOOR CHART" section through the end of the provided text.
    """
    # Define the start marker for the section to extract
    start_marker = "## PATIENT DOOR CHART"
    
    # Find the position where the relevant section starts
    start_index = text.find(start_marker)
    
    # If the section is found, extract and return the text from that point onwards
    if start_index != -1:
        return text[start_index:]
    else:
        # Return a message indicating the section was not found if it doesn't exist in the string
        return "PATIENT DOOR CHART section not found in the provided text."
# st.write(f'Here is the case {st.session_state.final_case}')

try:
    extracted_section = extract_patient_door_chart_section(st.session_state.final_case)
    st.info(extracted_section)
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": f'{sim_persona} Here are the specifics for your persona: {st.session_state.final_case}'}, ]
except Exception as e:
    st.error(f"Please return to the main page. An error occurred. Please do not 're-load' when in the middle of a conversation. Here are the error details: {e}. ")

if "sex" not in st.session_state:
    st.session_state.sex = ""
if st.session_state.sex == "":
    messages_sex =[{"role": "user", "content": f'Analyze the following content and return only the sex, e.g., male, female, or other. Return nothing else. {extracted_section}'}]
    st.session_state.sex = llm_call("anthropic/claude-3-haiku", messages_sex)

#################################################

# Set OpenAI API key from Streamlit secrets
groq_client = Groq(api_key = st.secrets['GROQ_API_KEY'])

# st.set_page_config(
#     page_title='Fast Helpful Chat',
#     page_icon='🌌',
#     initial_sidebar_state='expanded'
# )


st.title("Clinical Simulator Chat")
# st.caption('Powered by [Groq](https://groq.com/).')


if check_password():
    st.info("Type your questions at the bottom of the page or use voice input! You may need to right click your Chrome browser tab to unmute this website and also accept the microphone permissions.")

    
    st.sidebar.title('Customization')
    st.session_state.model = st.sidebar.selectbox(
            'Choose a model',
            ['llama3-70b-8192', 'llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
        )
        # Initialize chat history

        
    # if st.sidebar.checkbox("Change personality? (Will clear history.)"):
    #     persona = st.sidebar.radio("Pick the persona", ("Regular user", "Physician"), index=1)
    #     if persona == "Regular user":
    #         system = st.sidebar.text_area("Make your own system prompt or use as is:", value=system_prompt2)
    #     else:
    #         system = system_prompt
    #     st.session_state.messages = [{"role": "system", "content": system}]
        
    if "sim_response" not in st.session_state:
        st.session_state["sim_response"] = ""

    if "audio_off" not in st.session_state:
        st.session_state["audio_off"] = False

    if "audio_input" not in st.session_state:
        st.session_state["audio_input"] = ""
        
    if "voice" not in st.session_state:
        # Example usage:
        st.session_state["voice"] = assign_random_voice(st.session_state.sex)
        
    if "results" not in st.session_state:
        st.session_state["results"] = ""
        
    if "orders_placed" not in st.session_state:
        st.session_state["orders_placed"] = ""
        
    if "conversation_string" not in st.session_state:
        st.session_state["conversation_string"] = ""
        
    if "assessment" not in st.session_state:
        st.session_state["assessment"] = ""
        
    if "last_audio_size" not in st.session_state:
        st.session_state["last_audio_size"] = 0

            # Audio selection
    
    input_source = st.radio("Choose to type or speak!", ("Text", "Microphone"), index=0)
    st.session_state.audio_off = st.checkbox("Turn off voice generation", value=False) 
    # Display chat messages from history on app rerun
    conversation_str = extracted_section + "\n\n" + "______" + "\n\n" + "**Clinical Interview:**\n\n"
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message(message["role"], avatar="👩‍⚕️"):
                st.markdown(message["content"])
                conversation_str += "👩‍⚕️: " + message["content"] + "\n\n"
        elif message["role"] == "assistant":
            with st.chat_message(message["role"], avatar="🤒"):
                st.markdown(message["content"])
                conversation_str += "🤒: " + message["content"] + "\n\n"
    conversation_str += "______" + "\n\n" + "**Orders:**\n\n" + st.session_state.orders_placed +  "**Results:**\n\n""\n\n" + st.session_state.results + "\n\n"
    st.session_state.conversation_string = conversation_str



    if input_source == "Text":
    
    # Accept user input
        if prompt := st.chat_input("What's up?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user", avatar="👩‍⚕️"):
                st.markdown(prompt)
                
                # Display assistant response in chat message container
            with st.chat_message("assistant", avatar="🤒"):        
                stream = groq_client.chat.completions.create(
                    model=st.session_state["model"],
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    temperature=0.3,
                    stream=True,
                )
                st.session_state.sim_response = st.write_stream(parse_groq_stream(stream))
                
            st.session_state.messages.append({"role": "assistant", "content": st.session_state.sim_response})
    else:
        with st.sidebar:
            st.info("Click the green person-icon, pause 3 seconds, and begin to speak with natural speech.\
                    As soon as you pause, the LLM will start its response.")
            audio_bytes = audio_recorder(
                text="Click, pause, speak:",
                recording_color="#e8b62c",
                neutral_color="#6aa36f",
                icon_name="user",
                icon_size="3x",
            )

        if audio_bytes:
            try:
                # Save audio bytes to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    temp_file.write(audio_bytes)
                    audio_file_path = temp_file.name
                    
                                # Inform about audio file size
                file_stats = os.stat(audio_file_path)
                # st.write("We have audio bytes!!! Length: ", file_stats.st_size)
                if st.session_state["last_audio_size"] != file_stats.st_size:

                    with st.spinner("Transcribing audio... Please wait."):
                        prompt = transcribe_audio(audio_file_path)

                    st.session_state.messages.append({"role": "user", "content": prompt})

                    # Display user message in chat message container
                    with st.chat_message("user", avatar="👩‍⚕️"):
                        st.markdown(prompt)
                        
                    



            finally:
                # Ensure the tempfile is removed regardless of success or failure in processing
                if 'audio_file_path' in locals():
                    os.remove(audio_file_path)
                    # st.write("Temporary audio file removed.")

            # Clearing audio bytes manually, might be redundant if no other operations store this variable
            audio_bytes = None

            if st.session_state["last_audio_size"] != file_stats.st_size:    
                # Display assistant response in chat message container
                with st.chat_message("assistant", avatar="🤒"):
                    with st.spinner("Answering... Please wait."):        
                        stream = groq_client.chat.completions.create(
                            model=st.session_state["model"],
                            messages=[
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages
                            ],
                            temperature=0.3,
                            stream=True,
                        )
                    st.session_state.sim_response = st.write_stream(parse_groq_stream(stream))
                    
                st.session_state.messages.append({"role": "assistant", "content": st.session_state.sim_response})
                st.session_state["last_audio_size"] = file_stats.st_size
                
    
    if st.session_state.audio_off == False:

        if st.session_state.sim_response:
            with st.spinner("Synthesizing audio... Please wait."):
                talk_stream("tts-1", st.session_state.voice, st.session_state.sim_response)
            autoplay_local_audio("last_interviewer.mp3")
            st.info("Note - this is an AI synthesized voice.")            
            st.session_state.sim_response = "" 
            os.remove("last_interviewer.mp3")   
                

    # if st.session_state["sim_response"]:
    #     conversation_str = ""
    #     for message in st.session_state.messages:
    #         if message["role"] == "user":
    #             conversation_str += "👩‍⚕️: " + message["content"] + "\n\n"
    #         elif message["role"] == "assistant":
    #             conversation_str += "🤒: " + message["content"] + "\n\n"
    #     st.session_state.conversation_string = conversation_str
    #     html = markdown2.markdown(conversation_str, extras=["tables"])
    #     st.download_button('Download the conversation when done!', html, f'sim_response.html', 'text/html')
    #     st.session_state.sim_response = ""
    
    orders = st.sidebar.checkbox("Write Orders", value=False)
    if orders:
        with st.sidebar:
            order_details = st.text_input("Orders", key="order")

            if st.button("Submit Orders"):
                st.session_state.orders_placed = order_details + "\n\n" + st.session_state.orders_placed
                prompt = orders_prompt.format(order_details=order_details, case_details=st.session_state.final_case)
                orders_messages = [{"role": "user", "content": prompt}]
                with st.spinner("Transmitting Orders... Please wait."):
                    orders_results = llm_call("anthropic/claude-3-sonnet", orders_messages)
                st.session_state.results = orders_results['choices'][0]['message']['content'] + "\n\n" + st.session_state.results
            
            with st.expander("Prior Orders", expanded = False):                
                st.write(st.session_state.orders_placed)
            with st.expander("All Results", expanded = False):
                st.write(st.session_state.results)

    
    html2 = markdown2.markdown(st.session_state.conversation_string, extras=["tables"])
    st.sidebar.download_button('Download the transcript!', html2, f'transcript.html', 'text/html')
    st.sidebar.divider()         
    assess = st.sidebar.checkbox("Assess Interaction", value=False)
    
    if assess:
        student_level = st.sidebar.selectbox("Student Level", ["1st Year Medical Student", "2nd Year Medical Student", "3rd Year Medical Student", "4th Year Medical Student"])
        prompt = assessment_prompt.format(student_level = student_level, case_details=st.session_state.final_case, conversation_transcript=st.session_state.conversation_string, orders_placed=st.session_state.orders_placed, results=st.session_state.results)
        assessment_messages = [{"role": "user", "content": prompt}]
        if st.sidebar.button("Formulate Assessment"):
            with st.sidebar:
                with st.spinner("Formulating Assessment... Please wait."):
                    try:
                        assessment_response = llm_call("anthropic/claude-3-sonnet", assessment_messages)
                    except Exception as e:
                        st.error("Error formulating assessment, be sure to download the transcript and try again. Here are the error details: " + str(e))
            st.session_state.assessment = assessment_response['choices'][0]['message']['content']
        
        if st.session_state.assessment:
            with st.expander("Assessment", expanded = False):
                st.write(st.session_state.assessment)
            html = markdown2.markdown(st.session_state.assessment, extras=["tables"])
            st.sidebar.download_button('Download the assessment when done!', html, f'assessment.html', 'text/html')
        
        