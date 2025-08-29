# Reddit Sentiment & Emotion Analyzer

This project analyzes **sentiment** (positive/negative/neutral) and **emotion** (anger, joy, sadness, etc.) from Reddit posts using Hugging Face transformers.  
Visualizations include sentiment distribution, emotion distribution, score comparisons, and most common words.

---

## Features
- Fetches posts from any subreddit using **PRAW (Python Reddit API Wrapper)**.
- Runs **sentiment analysis** with DistilBERT.
- Runs **emotion analysis** with a DistilRoBERTa model.
- Generates **visualizations** (bar charts, boxplots, word frequency).
- Saves results to CSV.

---

## Tech Stack
- Python 3.8+
- [PRAW](https://praw.readthedocs.io/) – Reddit API Wrapper
- [Transformers (Hugging Face)](https://huggingface.co/transformers/)
- [Matplotlib](https://matplotlib.org/) – Data visualization
- [Pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)

---

##Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/dhirajkk91/Reddit-Sentiment-.git
   cd Reddit-Sentiment-
