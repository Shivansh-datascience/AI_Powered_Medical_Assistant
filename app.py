import streamlit as st
from google.cloud import storage
from vertexai.preview.generative_models import ChatSession, GenerativeModel
from pymongo import MongoClient as Mongo_db_client
from dotenv import load_dotenv
import os
import logging

from google.cloud import aiplatform

aiplatform.init(
    project="organic-storm-450000-r9",  
    location="us-central1"               
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class AI_Medical_Assistant:

    def __init__(self, user_input, file_input):
        self.user_input = user_input
        self.model = "gemini-2.5-flash"
        self.cloud_project = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.cloud_bucket_name = os.getenv("GOOGLE_CLOUD_BUCKET")  
        self.cloud_project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        self.custom_api_key = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
        self.GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
        self.file_input = file_input
        self.api_key = os.getenv("GOOGLE_GEMINI_MODEL_API_KEY")
        self.blob_name = os.getenv("GOOGLE_CLOUD_BLOB_NAME")
        self.vertex_ai = os.getenv("GOOGLE_GEN_AI_USE_VERTEX")
        self.gemini_model = os.getenv("GOOGLE_GEMINI_MODEL_API_KEY")

    def store_user_input_file_into_cloud_storage(self):
        logging.info("Storing User input file into cloud storage")
        try:
            storage_client = storage.Client(project=self.cloud_project_id)
            bucket = storage_client.bucket(self.cloud_bucket_name)
            blob = bucket.blob(self.blob_name)

            if hasattr(self.file_input,'read'):
                blob.upload_from_file(self.file_input)
            else:
                blob.upload_from_filename(self.file_input)
            logging.info(f"File uploaded to Cloud storage: {self.file_input}")
            return True
        except FileNotFoundError as file_not_found_exception:
            logging.error(f"{file_not_found_exception}")
            return False
        except FileExistsError as file_exist_error:
            logging.error(f"{file_exist_error}")
            return False
        except Exception as e:
            logging.error(f"{e}")
            return False

    def create_prompt_for_image_user_input(self):
        image_prompt = """
        You are an AI-powered medical assistant designed to process and extract relevant information from medical files. Your primary task is to analyze the provided file input, identify key medical details, and present them in a clear and concise manner.
        You will be provided with a file input containing medical information:
        {file_input}

        Follow these steps to process the file:
        1. Analyze the file to identify the type of medical information it contains (e.g., patient history, lab results, doctor's notes).
        2. Extract key details such as patient name, medical conditions, medications, and any relevant findings or observations.
        3. Summarize the extracted information in a structured format.
        4. If the file format is unsupported or the file does not contain relevant medical information, respond with an appropriate error message.

        Output the extracted information in the following format:
        - Patient Name: [patient name]
        - Medical Conditions: [list of conditions]
        - Medications: [list of medications]
        - Key Findings: [summary of findings]

        Ensure that the output is accurate, concise, and easy to understand for medical professionals."""
        return image_prompt

    def create_prompt_for_text_user_input(self):
        text_prompt = """
        You are a knowledgeable and helpful medical assistant. Your task is to process user input related to medical inquiries and provide appropriate responses.
        You will be provided with user input:
        {user_input}

        Follow these steps:
        1. Carefully analyze the user input to understand the medical question or request.
        2. Use your knowledge base to generate an accurate and informative response.
        3. If the input is unclear or requires more information, ask clarifying questions.
        4. Ensure your response is professional, empathetic, and easy to understand.
        5. If you cannot answer the question or if the input is inappropriate, provide a polite and informative message.
        Ensure that the output is accurate, concise, and easy to understand for medical professionals."""
        return text_prompt

    def create_response_for_image_input_user(self):
        logging.info("Generating response to Image input user")
        try:
            image_gemini_model = GenerativeModel(
                model_name=self.model,
                system_instruction=self.create_prompt_for_image_user_input()
            )
            image_response = image_gemini_model.generate_content([self.user_input, self.file_input])
            return image_response.text
        except Exception as e:
            logging.error(f"{e}")
            return None

    def create_response_for_user_input(self):
        logging.info("Generating user response")
        try:
            user_gemini_model = GenerativeModel(
                model_name=self.model,
                system_instruction=self.create_prompt_for_text_user_input()
            )
            user_response = user_gemini_model.generate_content([self.user_input])

            return user_response.text
        except Exception as e:
            logging.error(f"{e}")
            return None
        

import streamlit as st

# App title and header
st.set_page_config(page_title="AI Medical Assistant", page_icon="🩺")
st.title("🩺 Welcome to the AI Medical Assistant Bot")

# Subtitle
st.markdown("""
This assistant helps analyze medical images and provides you with a smart, concise AI-powered interpretation.

📁 **Step 1**: Upload a medical image (X-ray, MRI, CT, etc.)  
📝 **Step 2**: Optionally enter any symptoms or context  
🤖 **Step 3**: Receive a detailed response powered by Gemini  
""")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your medical image", type=["jpg", "jpeg", "png", "bmp", "tiff"])

# Optional user input
user_context = st.text_area("💬 Describe symptoms or ask a question (optional):", height=100)

# Analyze button
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("🧠 Analyze with AI Assistant"):
        assistant = AI_Medical_Assistant(user_input=user_context,file_input=uploaded_file)

        image_response = assistant.create_response_for_image_input_user()


        st.info("⏳ Analyzing image with Gemini model. Please wait...")

        # Placeholder for actual backend processing
        st.success("✅ AI Assistant Response:")

        st.write(image_response)

        st.write("📝 *Here's where the Gemini-generated medical response will appear.*")

        st.warning("⚠️ This is not a diagnosis. Always consult a licensed medical professional.")

else:
    st.warning("📎 Please upload an image to begin.")

if user_context:
    if st.button("🧠 Analyze with AI Assistant"):

        #calling the class for above class
        assistant = AI_Medical_Assistant(user_input=user_context,file_input=uploaded_file)

        response = assistant.create_response_for_user_input()

        # Placeholder for actual backend processing
        st.success("✅ AI Assistant Response:")
        st.write(response)
        st.write("📝 *Here's where the Gemini-generated medical response will appear.*")
        st.warning("⚠️ This is not a diagnosis. Always consult a licensed medical professional.")

# Footer
st.markdown("---")
st.markdown("👨‍⚕️ Built with [Google Vertex AI Gemini](https://cloud.google.com/vertex-ai) + [Streamlit](https://streamlit.io)")


