import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai
from openai import OpenAI
import streamlit.components.v1 as components


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
    challenges = data["What challenges are you currently facing in your role?"].dropna().tolist()
    suggestions = data["What suggestions do you have for improving team efficiency or project management?"].dropna().tolist()

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
            '<span style="color: red">**To Print this report, press CMD+P on Mac or Ctrl+P on Windows. For better visuals, select Landscape Layout.**</span>', 
            unsafe_allow_html=True
        )



        

if __name__ == "__main__":
    main()
