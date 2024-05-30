import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai
from openai import OpenAI
import streamlit.components.v1 as components
from PIL import Image
import base64


st.set_page_config(
    page_title=" PM - Feedback Analysis Tool ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
api_key = st.secrets["AI_KEY"]

# Initialize OpenAI client
client = OpenAI(api_key=api_key)


def summarize_sentiments(texts):
    """
    Summarize the overall sentiment of a collection of statements.
    """
    # Combine all texts into a single string for analysis
    combined_text = ". ".join(texts)

    # Send the combined text for sentiment analysis
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please summarize the overall sentiment of the following text: {combined_text}"}
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

    # Create a pie chart with a smaller size
    plt.figure(figsize=(4, 4), facecolor='none')  # Reduced figure size
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


# Function to suggest next steps for the project manager
def suggest_next_steps(data):
    """
    Analyze the data and suggest next steps for the project manager.
    """
    # Analyze sentiments of challenges and suggestions columns
    challenges = data["What challenges are you currently facing in your role?"].dropna(
    ).tolist()
    suggestions = data["What suggestions do you have for improving team efficiency or project management?"].dropna(
    ).tolist()

    # Combine the lists for sentiment analysis
    combined_text = " ".join(challenges + suggestions)

    # Get sentiment analysis from OpenAI (assuming the API client is already configured)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please analyze the following feedback and suggest next steps for the project manager: {combined_text}"}
        ]
    )
    # Extract and return the suggestion from the response
    return completion.choices[0].message.content

# Function to return JavaScript for screenshot


def main():

    banner_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Deckers_Outdoor_Corporation_201x_logo.svg/1599px-Deckers_Outdoor_Corporation_201x_logo.svg.png?20191005032104"
    centered_image_html = f"<div style='text-align: center'><img src='{banner_image_url}' width='300'></div><br>"
    st.markdown(centered_image_html, unsafe_allow_html=True)

    # Your app description or introduction
    st.markdown("<h1 style='text-align: center; color: #5F7E94;'> Feedback Sentiment Analysis Tool</h1>",
                unsafe_allow_html=True)

    st.markdown("<h5 style='text-align: center; color: #5F7E94;'> Welcome! Begin now to explore insights and recommendations to enhance team collaboration!</h5>",
                unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    col1, col2 = st.columns(2)

    # Access Survey Link
    with col1:
        st.subheader("Access Survey Link")
        st.write("")
        st.write("")
        with st.expander("Please choose a feedback survey"):
            link1 = "https://docs.google.com/forms/d/1_p6BvoJd4TkyM01UrvmgnVe5RUwKyWdeQhMJHvdxy9s/edit"
            link2 = "https://docs.google.com/forms/d/1q2h9TkfDY4BKb8esIP3ERfVsoTKpQUsjo6kHx7B7ZDA/edit"
            link3 = "https://docs.google.com/forms/u/0/?ec=asw-forms-globalnav-goto&tgif=d"

            st.markdown(f"[Employee Survey]({link1})")
            st.markdown(f"[Client Survey]({link2})")
            st.markdown(f"[Create Your Own Survey]({link3})")
    # Upload Excel File for Analysis
    with col2:
        st.subheader("Upload Excel File for Analysis")
        uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx'])
        if uploaded_file is not None:
            data = load_data(uploaded_file)
            st.write(data)

            # Initialize a counter to keep track of the column to use
            column_counter = 0

            # Initialize columns outside the loop
            col1, col2 = st.columns([1, 1])  # Adjust the ratio as needed

            for column in data.columns:
                if data[column].dtype == 'object':
                    unique_values = data[column].dropna().unique()
                    if all(len(value.split()) < 3 for value in unique_values):
                        fig = plot_pie_chart(data, column)

                        # Alternate between the two columns
                        if column_counter % 2 == 0:
                            with col1:
                                st.pyplot(fig)
                        else:
                            with col2:
                                st.pyplot(fig)

                        # Increment the counter after each iteration
                        column_counter += 1

                    else:
                        st.markdown(f"**Sentiment Analysis for: {column}**")
                        # Collect all responses for the question
                        responses = data[column].dropna().tolist()
                        # Summarize sentiments
                        summary = summarize_sentiments(responses)
                        st.write(summary)

            # New Section: Suggestions for Next Steps for Project Manager
            st.markdown("**Next Steps for Project Manager:**")
            next_steps = suggest_next_steps(data)
            st.write(next_steps)

            st.markdown(
                '<span style="color: red">**To Print this report, press on three dots on top right**</span>',
                unsafe_allow_html=True
            )


if __name__ == "__main__":
    main()
