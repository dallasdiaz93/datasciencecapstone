import pandas as pd
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import streamlit as st
from pages import Resume


if not firebase_admin._apps:
    # Initialize Firebase Admin SDK
    cred = credentials.Certificate("key\capstone-fbfa1-firebase-adminsdk-xepxl-92bf1972f2.json")
    firebase_admin.initialize_app(cred)

# Initialize Firestore client
db = firestore.client()

# Function to add data to Firestore
def add_data_to_firestore(collection_name, data):
    # Add data to Firestore collection
    for index, row in data.iterrows():
        doc_ref = db.collection(collection_name).document()
        doc_ref.set(row.to_dict())

# Streamlit UI
st.title("Firebase Data Importer")

# Upload CSV file
csv_file = st.file_uploader("Upload CSV file", type=["csv"])

# Text input for Firestore collection name
collection_name = st.text_input("Enter Firestore collection name")

# Button to add data
if st.button("Add Data"):
    if csv_file is None:
        st.warning("Please upload a CSV file.")
    elif not collection_name:
        st.warning("Please enter Firestore collection name.")
    else:
        # Read CSV file into pandas DataFrame
        data = pd.read_csv(csv_file)

        # Add data to Firestore
        add_data_to_firestore(collection_name, data)
        st.success("Data added to Firestore successfully.")
