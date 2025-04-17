import streamlit as st
import requests
import io
import time
import logging
import traceback
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_flask_port():
    """Read the Flask server port from the file."""
    try:
        if os.path.exists('flask_port.txt'):
            with open('flask_port.txt', 'r') as f:
                return int(f.read().strip())
    except Exception as e:
        logger.error(f"Error reading port file: {e}")
    return 5000

def check_server_health(port):
    """Check if the Flask server is running and healthy."""
    try:
        response = requests.get(f"http://127.0.0.1:{port}/")
        return response.status_code == 200
    except:
        return False

st.title("Dementia Detection - Audio Classification Test")


port = get_flask_port()
api_url = f"http://127.0.0.1:{port}/predict"


if not check_server_health(port):
    st.error(f"⚠️ Flask server is not running on port {port}. Please start the server first.")
    st.stop()

st.write("Upload a WAV audio file to test the model.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    if st.button("Predict"):
        files = {"file": uploaded_file.getvalue()}
        

        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1} of {max_retries}")
                logger.info(f"Connecting to {api_url}")
                

                response = requests.post(api_url, files=files, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("Prediction successful!")
                    st.write("**Class:**", result.get("class_name", "Unknown"))
                    st.write("**Confidence:**", result.get("confidence", {}))
                    break
                else:
                    error_msg = f"API Error: {response.status_code}"
                    logger.error(error_msg)
                    logger.error(response.text)
                    st.error(error_msg)
                    st.error(response.text)
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    
            except requests.exceptions.ConnectionError as e:
                error_msg = f"Connection error: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                st.error("Connection error. Please ensure the API server is running.")
                st.error(error_msg)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                    
            except requests.exceptions.Timeout as e:
                error_msg = f"Timeout error: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                st.error("Request timed out. The server is taking too long to respond.")
                st.error(error_msg)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                    
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                st.error("Error while connecting to the API.")
                st.error(error_msg)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
