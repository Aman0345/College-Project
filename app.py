import asyncio
import streamlit as st
from google import genai
from google.genai import types
import os
from emotion_classifier.utils import PROMPT
import tempfile
import shutil


# Set up the event loop for asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

upload_dir = 'uploaded_files'
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)


def analyzeImage(up_file):
    with st.spinner("Uploading file"):
        path = os.path.join(upload_dir, up_file.name)

        with open(path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        myfile = client.files.upload(file=path)

        st.success("Done")

    with st.spinner("Analyzing the sentiment"):
        try:
            result = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    myfile,
                    "\n\n",
                    PROMPT],
                )
            st.markdown(result.text)
            
        except Exception as e:
            st.error(e)




def analyzeFile(up_file):
    with st.spinner("Uploading file"):
        path = os.path.join(upload_dir, up_file.name)
        with open(path, "wb") as tmp_file:
            tmp_file.write(up_file.getvalue())
        myfile = client.files.upload(file=path)
        
        st.success("Done")
    with st.spinner("Analyzing the sentiment"):
        try:
            result = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    myfile,
                    "\n\n",
                    PROMPT],
                )
            st.markdown(result.text)
            
        except Exception as e:
            st.error(e)





# Initialize GenAI client
client = genai.Client(api_key="AIzaSyAb46OomF6UQLpqZhB8JIwGxCpWerFC16I")

# Set page config with a custom theme (like ChatGPT's red and black)
st.set_page_config(page_title="Sentiment Analysis AI", page_icon="🧠", layout="wide")

# Title of the app
st.title('Sentiment Analysis AI')

# Main container for layout
with st.container():

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        
        text_input = st.text_area("Enter your text here", key="text_input")

        # Button for sentiment analysis
        if text_input and st.button("Analyze Sentiment"):
            # Call GenAI model for sentiment analysis
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash", 
                    contents=("analyze the sentiment of this text " + text_input)
                )
                # Display the result
                st.markdown("### Sentiment Analysis Result:")
                st.write(response.text)
            except Exception as e:
                st.error(f"Error analyzing sentiment: {e}")



        # for youtube video 
        link_input = st.chat_input("Enter video Link")
        if link_input:
            # Call GenAI model for sentiment analysis
            with st.spinner("Analyzing the video"):
                try:
                    response = client.models.generate_content(
                    model="gemini-2.0-flash", 
                    contents =[
            types.Content(
                role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=link_input,
                    mime_type="video/*",
                ),
                types.Part.from_text(text=PROMPT),
            ],
            ),
        ]
                )
                # Display the result
                    st.markdown("### Sentiment Analysis Result:")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Error analyzing sentiment: {e}")



        # Section for image upload (currently not integrated for analysis)
        uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image", width=300)
            analyzeImage(uploaded_image)
            st.write("Sentiment analysis for images is not yet integrated but can be added.")



        # Section for file upload (TXT, PDF, DOCX)
        uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])
        if uploaded_file is not None:
            analyzeFile(uploaded_file)
