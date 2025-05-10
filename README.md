
# Sentiment Analysis on Amazon Review Dataset 🛒🧠

This project uses Natural Language Processing (NLP) techniques to perform **sentiment analysis** on customer reviews from the **Amazon Review** dataset. The goal is to classify each review as either **positive** or **negative**, providing insights into customer satisfaction.

# 📁 Project Structure

* `sentiment_analysis_on_amazon_dataset_nlp.ipynb` — Main Jupyter notebook containing code and step-by-step explanations.
* `amazon_review.csv` — Dataset file used for training and testing (ensure it's available in your working directory).
* `README.md` — Project overview and setup guide.

# 🔍 Dataset

The **Amazon Review** dataset contains customer feedback with the following columns:

* `ReviewText` — Text content of the customer review
* `Sentiment` — Binary label: `1` for positive, `0` for negative sentiment

#📌 Features

* Text preprocessing: lowercasing, tokenization, stopword removal, stemming/lemmatization
* Feature extraction using TF-IDF
* Classification using ML models (e.g., Logistic Regression)
* Evaluation metrics: Accuracy, Precision, Recall, F1 Score

#🛠️ Technologies Used

* Python 3.x
* Jupyter Notebook
* Scikit-learn
* NLTK / spaCy
* Pandas, NumPy
* Matplotlib, Seaborn

# 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/Tanujashree27/sentiment-analysis-amazon.git
   cd sentiment-analysis-amazon
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure the dataset file `amazon_review.csv` is in the same directory as the notebook.

4. Run the notebook:

   ```bash
   jupyter notebook sentiment_analysis_on_amazon_dataset_nlp.ipynb
   ```

#📊 Results

The sentiment classifier achieved an accuracy of approximately **XX%** (replace with actual result). The model performs well in identifying sentiment polarity in customer reviews.

# ✅ Future Improvements

* Integrate deep learning models (e.g., LSTM, BERT)
* Expand to neutral sentiment classification
* Deploy as a web or API-based sentiment analyzer

# 🤝 Contributing

Contributions are welcome! Please fork this repository and open a pull request.



