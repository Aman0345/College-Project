import asyncio
import streamlit as st
from google import genai

# Set up the event loop for asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

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

        # Section for image upload (currently not integrated for analysis)
        uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
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
