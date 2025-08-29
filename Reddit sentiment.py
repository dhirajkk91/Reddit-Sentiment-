import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
import praw.exceptions
import requests.exceptions
import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter
import re


import praw
reddit = praw.Reddit(
    client_id="eXHHZI-7VtG9RHb2nC6Gug",
    client_secret="UqX81wlan36NuFEvkQ_fhTCSfw30Lw",
    user_agent="test by /u/Professional_Split21"
)


def fetch_reddit_posts(reddit_instance, subreddit_name="wallstreetbets", limit=10):
    """
    Fetches a specified number of hot posts from a given subreddit with error handling.

    Args:
        reddit_instance: An authenticated PRAW Reddit instance.
        subreddit_name (str): The name of the subreddit to fetch posts from.
        limit (int): The maximum number of posts to fetch.

    Returns:
        pd.DataFrame: A DataFrame containing the title, selftext, score, and url of the fetched posts, or an empty DataFrame in case of error.
    """
    posts = []
    try:
        print(f"Attempting to fetch posts from r/{subreddit_name}...")
        for submission in reddit_instance.subreddit(subreddit_name).hot(limit=limit):
            posts.append({
                "title": submission.title,
                "selftext": submission.selftext,
                "score": submission.score,
                "url": submission.url
            })
        print(f"Successfully fetched {len(posts)} posts.")
        return pd.DataFrame(posts)
    except praw.exceptions.APIException as e:
        print(f"Reddit API Error fetching posts: {e}")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"Network Error fetching posts: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred during post fetching: {e}")
        return pd.DataFrame()


def analyze_sentiment(text, sentiment_model):
    """
    Analyzes the sentiment of a given text using a pre-trained sentiment model with error handling.

    Args:
        text (str): The input text string.
        sentiment_model: A Hugging Face transformers pipeline for sentiment analysis.

    Returns:
        dict: A dictionary with the sentiment label and score. Returns {"label": "NEUTRAL", "score": 0.0} for empty or whitespace-only text or if model is not loaded. Returns {"label": "ERROR", "score": 0.0} on analysis error.
    """
    if sentiment_model is None:
        return {"label": "MODEL_ERROR", "score": 0.0}
    if not isinstance(text, str) or not text.strip():
        return {"label": "NEUTRAL", "score": 0.0}
    try:
        # Truncate text to 512 tokens as required by the model
        result = sentiment_model(text[:512])[0]
        return result
    except Exception as e:
        # Log the error but continue processing other texts
        print(f"Error analyzing sentiment for text: {text[:10]}... - {e}")
        return {"label": "ERROR", "score": 0.0}


def analyze_emotion(text, emotion_model):
    """
    Analyzes the emotion of a given text using a pre-trained emotion model with error handling.

    Args:
        text (str): The input text string.
        emotion_model: A Hugging Face transformers pipeline for emotion analysis.

    Returns:
        dict: A dictionary with the emotion label and score. Returns {"label": "NEUTRAL", "score": 0.0} for empty or whitespace-only text or if model is not loaded. Returns {"label": "ERROR", "score": 0.0} on analysis error.
    """
    if emotion_model is None:
        return {"label": "MODEL_ERROR", "score": 0.0}
    if not isinstance(text, str) or not text.strip():
        return {"label": "NEUTRAL", "score": 0.0}
    try:
        # Truncate text to 512 tokens as required by the model
        result = emotion_model(text[:512])[0]
        return result
    except Exception as e:
        # Log the error but continue processing other texts
        print(f"Error analyzing emotion for text: {text[:10]}... - {e}")
        return {"label": "ERROR", "score": 0.0}


def visualize_sentiment(df_sentiment):
    """
    Generates a bar plot showing the distribution of sentiment labels with error handling.

    Args:
        df_sentiment (pd.DataFrame): DataFrame containing a "sentiment" column.
    """
    if df_sentiment.empty or "sentiment" not in df_sentiment.columns:
        print("DataFrame is empty or missing 'sentiment' column for sentiment visualization.")
        return

    sentiment_counts = df_sentiment["sentiment"].value_counts()

    try:
        plt.figure(figsize=(6,4))
        sentiment_counts.plot(kind="bar", color=["green", "red", "blue"])
        plt.title("Sentiment Distribution of Reddit Posts")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        plt.xticks(rotation=0) # Keep labels horizontal
        plt.tight_layout() # Adjust layout to prevent labels overlapping
        plt.show()
    except Exception as e:
        print(f"Error generating sentiment visualization: {e}")


def visualize_emotion(df_emotion):
    """
    Generates a bar plot showing the distribution of emotion labels with error handling.

    Args:
        df_emotion (pd.DataFrame): DataFrame containing an "emotion" column.
    """
    if df_emotion.empty or "emotion" not in df_emotion.columns:
        print("DataFrame is empty or missing 'emotion' column for emotion visualization.")
        return

    emotion_counts = df_emotion["emotion"].value_counts()

    try:
        plt.figure(figsize=(10, 6))
        emotion_counts.plot(kind="bar", color=plt.cm.viridis(range(len(emotion_counts))))
        plt.title("Emotion Distribution of Reddit Posts")
        plt.xlabel("Emotion")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right") # Rotate labels for better readability
        plt.tight_layout() # Adjust layout to prevent labels overlapping
        plt.show()
    except Exception as e:
        print(f"Error generating emotion visualization: {e}")


def visualize_score_by_sentiment(df_score_sentiment):
    """
    Generates a box plot showing the distribution of scores by sentiment with error handling.

    Args:
        df_score_sentiment (pd.DataFrame): DataFrame containing "score" and "sentiment" columns.
    """
    if df_score_sentiment.empty or "score" not in df_score_sentiment.columns or "sentiment" not in df_score_sentiment.columns:
        print("Skipping score distribution visualization: DataFrame is empty or missing required columns.")
        return

    try:
        plt.figure(figsize=(8, 6))
        df_score_sentiment.boxplot(column="score", by="sentiment")
        plt.title("Distribution of Scores by Sentiment")
        plt.suptitle("") # Suppress the default boxplot title
        plt.xlabel("Sentiment")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error generating score distribution visualization: {e}")


def clean_text(text):
    """
    Cleans text by lowercasing and removing non-alphabetic characters.

    Args:
        text (str): The input text string.

    Returns:
        str: The cleaned text. Returns "" for non-string inputs.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove non-alphabetic characters
    return text


def get_most_common_words(text_series, n=10):
    """
    Finds the most common words in a pandas Series of text.

    Args:
        text_series (pd.Series): A Series containing text strings.
        n (int): The number of most common words to return.

    Returns:
        list: A list of tuples (word, count) for the most common words, or an empty list if the input series is empty.
    """
    if text_series.empty:
        return []
    cleaned_text = text_series.apply(clean_text).str.cat(sep=" ")
    words = cleaned_text.split()
    # Filter out stop words and short words
    filtered_words = [word for word in words if word not in ENGLISH_STOP_WORDS and len(word) > 2]
    return Counter(filtered_words).most_common(n)


def plot_most_common_words(word_list, sentiment_label):
    """
    Generates a bar plot for the most common words with error handling.

    Args:
        word_list (list): A list of tuples (word, count).
        sentiment_label (str): The sentiment category label.
    """
    if not word_list:
        print(f"No words to plot for {sentiment_label} sentiment.")
        return
    words, counts = zip(*word_list)
    try:
        plt.figure(figsize=(10, 6))
        plt.bar(words, counts, color=plt.cm.plasma(range(len(words))))
        plt.title(f"Most Common Words in {sentiment_label} Reddit Post Titles")
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error generating common words plot for {sentiment_label}: {e}")


def save_results(df_results, filename="reddit_sentiment_results.csv"):
    """
    Saves the DataFrame to a CSV file with error handling.

    Args:
        df_results (pd.DataFrame): The DataFrame to save.
        filename (str): The name of the CSV file.
    """
    if df_results.empty:
        print("DataFrame is empty, not saving.")
        return
    try:
        df_results.to_csv(filename, index=False)
        print(f" Analysis complete! Results saved as {filename}")
    except OSError as e:
        print(f"Error saving results to {filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving results: {e}")


# --- Main Execution ---

if __name__ == "__main__":
    # Load models with error handling
    try:
        sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        print(" Sentiment analysis model loaded successfully.")
    except Exception as e:
        print(f"Error loading sentiment model: {e}")
        sentiment_model = None  # Set to None if initialization fails

    try:
        emotion_model = pipeline("sentiment-analysis", model="j-hartmann/emotion-english-distilroberta-base")
        print(" Emotion analysis model loaded successfully.")
    except Exception as e:
        print(f"Error loading emotion model: {e}")
        emotion_model = None

    # Fetch Reddit Posts
    # Replace 'reddit' with your authenticated PRAW instance
    df = fetch_reddit_posts(reddit, subreddit_name="wallstreetbets", limit=30)

    if not df.empty:
        # Perform Sentiment Analysis
        if sentiment_model:
            sentiment_results = df["title"].apply(lambda x: analyze_sentiment(x, sentiment_model))
            df["sentiment"] = sentiment_results.apply(lambda x: x["label"])
            df["confidence"] = sentiment_results.apply(lambda x: x["score"])
        else:
            print("Sentiment model not available. Skipping sentiment analysis.")
            df["sentiment"] = "MODEL_ERROR"
            df["confidence"] = 0.0

        # Perform Emotion Analysis
        if emotion_model:
            emotion_results = df["title"].apply(lambda x: analyze_emotion(x, emotion_model))
            df["emotion"] = emotion_results.apply(lambda x: x["label"])
            df["emotion_score"] = emotion_results.apply(lambda x: x["score"])
        else:
            print("Emotion model not available. Skipping emotion analysis.")
            df["emotion"] = "MODEL_ERROR"
            df["emotion_score"] = 0.0

        # Visualize Results
        visualize_sentiment(df)
        visualize_emotion(df)
        visualize_score_by_sentiment(df)

        # Common words analysis and visualization
        if "title" in df.columns and "sentiment" in df.columns:
            positive_titles = df[df["sentiment"] == "POSITIVE"]["title"]
            negative_titles = df[df["sentiment"] == "NEGATIVE"]["title"]

            most_common_positive_words = get_most_common_words(positive_titles)
            most_common_negative_words = get_most_common_words(negative_titles)

            print("\nMost common words in POSITIVE titles:")
            print(most_common_positive_words)

            print("\nMost common words in NEGATIVE titles:")
            print(most_common_negative_words)

            plot_most_common_words(most_common_positive_words, "POSITIVE")
            plot_most_common_words(most_common_negative_words, "NEGATIVE")
        else:
            print("Skipping common words analysis: DataFrame is empty or missing required columns.")

        # Save Results
        save_results(df, filename="reddit_sentiment_results.csv")

        # Display head of the dataframe
        print(df.head(10))
    else:
        print("No data fetched from Reddit. Skipping analysis, visualization, and saving.")