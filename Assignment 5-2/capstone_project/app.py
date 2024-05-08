import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
#from openai import OpenAI
import plotly.graph_objs as go
import geopandas as gpd
import plotly.express as px
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.directory import DirectoryLoader
import os




#world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))



# Set up OpenAI API key

openai_api_key = os.getenv("OPENAI_API_KEY", "")

# Get your loader ready
loader = DirectoryLoader('content/', glob='**/*.txt')

# Load up your text into documents
documents = loader.load()

# Get your text splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Split your documents into texts
texts = text_splitter.split_documents(documents)

# Turn your texts into embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Get your docsearch ready
docsearch = FAISS.from_documents(texts, embeddings)

# Load up your LLM
llm = OpenAI(openai_api_key=openai_api_key)

# Create the RetrievalQA instance
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)



layoff_data = pd.read_csv("content/tweets_clean.csv")
# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("content/layoffs_clean.csv")  # Replace "your_dataset.csv" with your actual dataset filename
    data['date'] = pd.to_datetime(data['date'])  # Convert 'date' column to datetime format
    return data




def main():
    st.title("Interactive Analysis of Layoffs")

    
    # Load the dataset
    data = load_data()
    
    # Filter by country or location
    filter_by = st.selectbox("Filter by", ["Country", "Location"])

    if filter_by == "Country":
        selected_filter = st.selectbox("Select Country", data['country'].unique())
        filtered_data = data[data['country'] == selected_filter]
    else:
        selected_filter = st.selectbox("Select Location", data['location'].unique())
        filtered_data = data[data['location'] == selected_filter]

    # Create a two-column layout for start date and end date selection
    col1, col2 = st.columns(2)

    # Date range inputs
    with col1:
        start_date = st.date_input("Start Date")

    with col2:
        end_date = st.date_input("End Date")

    # Convert date inputs to datetime objects
    start_date = datetime.combine(start_date, datetime.min.time())
    end_date = datetime.combine(end_date, datetime.max.time())

        
    # Filter data by date range
    filtered_data = filtered_data[(filtered_data['date'] >= start_date) & (filtered_data['date'] <= end_date)]
    
    if filter_by == "Country":
        # Get top 15 locations
        top_locations = filtered_data['location'].value_counts().nlargest(15).index.tolist()

        # Filter data by top 15 locations
        filtered_data = filtered_data[filtered_data['location'].isin(top_locations)]

        # Group by country and sum up the total laid off
        country_total_laid_off = filtered_data.groupby('country')['total_laid_off'].sum().reset_index()

        # Plot choropleth map using Plotly
        fig = px.choropleth(country_total_laid_off, 
                            locations='country', 
                            locationmode='country names',
                            color='total_laid_off',
                            color_continuous_scale=px.colors.sequential.Plasma,
                            title='Total Laid Off by Country')
        
        # Customize map layout
        fig.update_layout(
            geo=dict(
                bgcolor='rgba(135,206,235,1)',  # Set background color (here, transparent)
                projection_scale=1.5,  # Adjust zoom level
            )
        )
        # Set the layout options
        fig.update_layout(geo=dict(showcoastlines=True))
        st.plotly_chart(fig)
        # Calculate total people laid off with commas for thousand separators
        total_laid_off = int(filtered_data['total_laid_off'].sum())
        total_laid_off_text = f"Total People Laid Off in {selected_filter}: {total_laid_off:,}"  # Add commas for thousand separators

        # Display total people laid off
        st.write(total_laid_off_text)

        # Plot total laid off by location and donut chart side by side
        st.subheader(f"Total Laid Off by Location (Top 15) for {selected_filter}")

        # Create a two-column layout for plots
        col1, col2 = st.columns(2)

        # Plot total laid off by location
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))  # Adjust size of the plot
            sns.barplot(x='location', y='total_laid_off', data=filtered_data, ax=ax, palette='viridis')
            ax.set_xlabel("Location")
            ax.set_ylabel("Total Laid Off")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_title(f"Total Laid Off by Location (Top 15) for {selected_filter}")
            plt.tight_layout()
            st.pyplot(fig)

        # Plot donut chart for total laid off by location
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))  # Adjust size of the plot
            location_counts = filtered_data['location'].value_counts()
            explode = [0.05] * len(location_counts)  # Add explosion for better separation
            ax.pie(location_counts, labels=None, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis', len(location_counts)), explode=explode)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax.set_title('Total Laid Off by Location (Top 15)')

            # Add legend with custom positioning
            plt.legend(location_counts.index, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

            plt.tight_layout()
            st.pyplot(fig)

        # Display total laid off as title text
        st.write("")
        st.write(f"### {total_laid_off_text}")

    else:
        # Plot total laid off by industry
        st.subheader(f"Total Laid Off by Industry for {selected_filter}")

        # Create a two-column layout for plots
        col1, col2 = st.columns(2)

        # Plot total laid off by industry
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))  # Adjust size of the plot
            sns.barplot(x='industry', y='total_laid_off', data=filtered_data, ax=ax, palette='viridis')
            ax.set_xlabel("Industry")
            ax.set_ylabel("Total Laid Off")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_title(f"Total Laid Off by Industry for {selected_filter}")
            plt.tight_layout()
            st.pyplot(fig)

        # Plot pie chart for total laid off by industry
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))  # Adjust size of the plot
            industry_counts = filtered_data['industry'].value_counts()
            explode = [0.05] * len(industry_counts)  # Add explosion for better separation
            ax.pie(industry_counts, labels=None, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis', len(industry_counts)), explode=explode)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax.set_title('Total Laid Off by Industry Distribution')

            # Add legend with custom positioning
            plt.legend(industry_counts.index, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

            plt.tight_layout()
            st.pyplot(fig)

    
        
# Lay off Stories analysis section
st.header("Lay off Stories analysis")



    # Define analysis topics
topics = ["5 Common Themes of layoff stories", "Leadership communication gap with layoff employees", "Support Resources layoff people should consider", "Areas of improvement for higher management for the employees layoff"]

# Function to generate analysis based on selected topic
def generate_analysis(topic):
    # Invoke the question-answering model with the selected topic
    result = qa.invoke({"query": f"give me analysis on {topic}"})
    return result['result']


# Define custom topic names for buttons
custom_topics = ["Common Themes", "Communication Gap", "Support Resources", "Improvement Areas"]

# Function to display buttons and handle button clicks
def display_buttons_and_handle_clicks():
    # Create a four-column layout for buttons
    col1, col2, col3, col4 = st.columns(4)

    # Buttons for different analysis topics
    button_clicked = False
    with col1:
        if st.button(custom_topics[0]):
            button_clicked = True
            response = generate_analysis(topics[0])
    if button_clicked:
        st.write(f"{custom_topics[0]} analysis:")
        st.write(response)

    button_clicked = False
    with col2:
        if st.button(custom_topics[1]):
            button_clicked = True
            response = generate_analysis(topics[1])
    if button_clicked:
        st.write(f"{custom_topics[1]} analysis:")
        st.write(response)

    button_clicked = False
    with col3:
        if st.button(custom_topics[2]):
            button_clicked = True
            response = generate_analysis(topics[2])
    if button_clicked:
        st.write(f"{custom_topics[2]} analysis:")
        st.write(response)

    button_clicked = False
    with col4:
        if st.button(custom_topics[3]):
            button_clicked = True
            response = generate_analysis(topics[3])
    if button_clicked:
        st.write(f"{custom_topics[3]} analysis:")
        st.write(response)

# Define the Streamlit app layout
#st.title("Layoff Story Analysis")

# Call the function to display buttons and handle button clicks
display_buttons_and_handle_clicks()

if __name__ == "__main__":
    main()
