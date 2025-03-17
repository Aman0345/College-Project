import asyncio
import streamlit as st
from google import genai
from google.genai import types
import os

# Set up the event loop for asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

upload_dir = 'uploaded_files'
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)


def analyzeImage(up_file):
    st.write("Analyzing image")
    myfile = client.files.upload(file=up_file)
    
    contents =[
            types.Content(
                role="user",
            parts=[
                types.Part.from_uri(
                    file_content = myfile.uri,
                    mime_type=myfile.mime_type,
                ),
                types.Part.from_text(text="""Analyze the sentiment of the video and tell the the sentiment only with a small description"""),
            ],
            ),
        ]
    try:
        result = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                myfile,
                "\n\n",
                "Can you tell me about the instruments in this photo?"],
            )
        st.markdown(result.text)
        
    except Exception as e:
        print(e)




# Initialize GenAI client
client = genai.Client(api_key="AIzaSyAb46OomF6UQLpqZhB8JIwGxCpWerFC16I")

# Set page config with a custom theme (like ChatGPT's red and black)
st.set_page_config(page_title="Sentiment Analysis AI", page_icon="🧠", layout="wide")

# Title of the app
st.title('Sentiment Analysis AI')

# Main container for layout
with st.container():
    # Create columns for better layout
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        # Section for text input
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
                    file_uri="https://youtu.be/jVMzoUkTXZ0",
                    mime_type="video/*",
                ),
                types.Part.from_text(text="""Analyze the sentiment of the video and tell the the sentiment only with a small description"""),
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
            path = os.path.join(upload_dir, uploaded_image.name)
            with open(path, "wb") as f:
                f.write(uploaded_image.getbuffer())
            st.write(path)
            analyzeImage(path)
            st.write("Sentiment analysis for images is not yet integrated but can be added.")

        # Section for file upload (TXT, PDF, DOCX)
        uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])
        
        if uploaded_file is not None:
            # Read content of the uploaded file (if it's a text file)
            if uploaded_file.type == "text/plain":
                file_content = uploaded_file.read().decode("utf-8")
                st.text_area("File Content", value=file_content, height=150)
                
                if st.button("Analyze Sentiment from File"):
                    # Placeholder for sentiment analysis result
                    st.write("Sentiment analysis for file content is not yet integrated.")
            else:
                st.write("Sentiment analysis for this file type is not yet integrated.")


