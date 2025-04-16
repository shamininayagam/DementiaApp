import streamlit as st
import requests
import io

st.title("Dementia Detection - Audio Classification Test")

st.write("Upload a WAV audio file to test the model.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    # Display the uploaded audio file
    st.audio(uploaded_file, format="audio/wav")
    
    if st.button("Predict"):
        # Prepare the file for the POST request
        # files = {"file": uploaded_file.getvalue()}
        # files={
        #     "file":(uploaded_file.name,io.BytesIO(uploaded_file.getvalue()),"audio/wav")
        # }
        files = {"file": (uploaded_file.name,uploaded_file,"audio/wave")}
    

        # URL of the Flask API endpoint (update if needed)
        api_url = "http://127.0.0.1:5000/predict"
        
        try:
            # response = requests.post(api_url, files={"file": uploaded_file})
            
            response = requests.post(api_url, files=files)

            if response.status_code == 200:
                result = response.json()
                st.success("Prediction successful!")
                st.write("**Class:**", result.get("class_name", "Unknown"))
                st.write("**Confidence:**", result.get("confidence", {}))
            else:
                st.error(f"API Error: {response.status_code}")
                st.error(response.text)
        except Exception as e:
            st.error("Error while connecting to the API.")
            st.error(str(e))
