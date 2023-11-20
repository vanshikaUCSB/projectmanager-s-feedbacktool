import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai
from openai import OpenAI


st.set_page_config(
    page_title="PM - Feedback Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
api_key = st.secrets["AI_KEY"]

# Initialize OpenAI client
client = OpenAI(api_key=api_key)


def summarize_sentiments(texts):
    """
    Summarize the sentiment of multiple statements using OpenAI's language model.
    """

    # Aggregate all texts into one for analysis
    combined_text = " ".join(texts)

    # Send the combined text for sentiment analysis
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please summarize the sentiment of the following statements: {combined_text}"}
        ]
    )

    # Extract the summary from the response
    return completion.choices[0].message.content



def analyze_sentiment(text):
    """
    Analyze the sentiment of the given text using OpenAI's language model.
    """

    # Replace with your actual API key

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"What is the sentiment of this statement? \"{text}\""}
        ]

    )

    # Assuming the response contains the sentiment
    return completion.choices[0].message.content


def load_data(file):
    """
    Load the Excel file into a Pandas DataFrame.
    """
    return pd.read_excel(file)


def plot_pie_chart(data, column):
    """
    Plot a pie chart for a given column of the DataFrame with some stylistic adjustments.
    """
    # Count the unique values in the column
    counts = data[column].value_counts()

    # Create a pie chart without a white background
    # Smaller figure size and no background
    plt.figure(figsize=(6, 6), facecolor='none')
    wedges, texts, autotexts = plt.pie(
        counts, labels=counts.index, autopct='%1.1f%%', startangle=140)

    # Improve the display of labels and percentages
    for text, autotext in zip(texts, autotexts):
        text.set_color('grey')  # Set the color of labels
        autotext.set_color('white')  # Set the color of percentages

    # Add a legend to the pie chart
    plt.legend(wedges, counts.index,
               title="Key",
               loc="center left",
               bbox_to_anchor=(1, 0, 0.5, 1))

    plt.title(column, color='grey')  # Set title with grey color
    return plt


def main():
    st.title("PM - Feedback Analysis Tool")
    
    # Your app description or introduction
    st.markdown("""
    This tool is designed to analyze and visualize participant feedback.
    Upload your Excel file, and the app will automatically generate pie charts
    for categorical data and provide sentiment analysis for open-ended responses.
    """)

    uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx'])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write(data)

        for column in data.columns:
            if data[column].dtype == 'object':
                unique_values = data[column].dropna().unique()
                if all(len(value.split()) < 3 for value in unique_values):
                    fig = plot_pie_chart(data, column)
                    st.pyplot(fig)
                else:
                    st.markdown(f"**Sentiment Analysis for: {column}**")
                    # Collect all responses for the question
                    responses = data[column].dropna().tolist()
                    # Summarize sentiments
                    summary = summarize_sentiments(responses)
                    st.write(summary)





if __name__ == "__main__":
    main()
