
import streamlit as st
from openai import OpenAI
import PyPDF2
from docx import Document
import os
# Import other necessary libraries

# Set up OpenAI API key
client = OpenAI(api_key="")


# Streamlit app title
st.title("Resume Analyzer")

# File upload section
resume_file = st.file_uploader("Upload Resume (PDF or DOCX)")

# Job description input
job_description = st.text_area("Enter Job Description")


# Function to extract text from uploaded file
def extract_text_from_file(file):
    # Check file type
    if file.type == "application/pdf":
        # Extract text from PDF
        text = read_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # Extract text from DOCX
        text = read_docx(file)
    else:
        st.error("Unsupported file format. Please upload a PDF or DOCX file.")
        return None
    return text

# Function to read text from PDF file
def read_pdf(file):
    text = ""
    pdf_reader = PyPDF2.PdfFileReader(file)
    for page_num in range(pdf_reader.numPages):
        text += pdf_reader.getPage(page_num).extractText()
    return text

# Function to read text from DOCX file
def read_docx(file):
    text = ""
    doc = Document(file)
    for paragraph in doc.paragraphs:
        text += paragraph.text
    return text

# # Function to generate analysis using GPT-3.5 API
def generate_analysis(resume_text, job_description):
    # Define messages including user questions and system responses
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Compare the candidate's qualifications with the job description and provide the overall percentage relevency score bolded in new line at the beginning and breif insights of the comparison."},
        {"role": "assistant", "content": f"{resume_text}\n\n{job_description}"},
    ]
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Choose the GPT-3.5 model
        messages=messages,
        max_tokens=1500,  # Adjust based on your desired length of response
        n=1,  # Number of responses to generate
        stop=None,  # Tokens to stop generation
        temperature=0.5,  # Control the randomness of the output
    )
    
    return response.choices[0].message.content.strip()



# Button to analyze resume
if st.button("Analyze Resume"):
    if resume_file is not None and job_description.strip() != "":
        # Perform analysis using GPT-3.5 API
        # Convert resume_file to text
        resume_text = extract_text_from_file(resume_file)
        # Generate analysis using GPT-3.5
        analysis = generate_analysis(resume_text, job_description)
        # Display analysis results
        st.write("## Resume Analysis")
        st.write(analysis)
    else:
        st.warning("Please upload a resume file and enter a job description.")




# Optionally, integrate LangChain for resume and job description verification
# Implement LangChain integration logic here

