# NLP

Sentiment analysis on airline tweets - classifying tweets as positive or negative and visualizing the most common words in each.

## Dataset

Used the Twitter US Airline Sentiment dataset (`Tweets.csv`) - 14,640 tweets directed at major US airlines, each labeled positive, negative, or neutral. Dropped the neutral tweets to keep this a binary classification problem, which left 9,178 negative and 2,363 positive tweets to work with. Originally sourced from [Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment).

## What's in here

Cleaned up the tweet text - lowercased everything, stripped out URLs, @mentions, hashtags, punctuation, numbers, and stopwords. Converted the cleaned text into TF-IDF vectors (top 5000 features) and trained two models on it:

- Logistic Regression
- Multinomial Naive Bayes

Checked accuracy and classification report for both. Also generated word clouds separately for positive and negative tweets to see which words show up most in each, and wrote a small function at the end to predict sentiment on any new sentence you type in.

## Results

| Model | Accuracy | Positive Recall | Positive Precision |
|---|---|---|---|
| Logistic Regression | 89.8% | 0.53 | 0.90 |
| Multinomial Naive Bayes | 88.1% | 0.40 | 0.95 |

Logistic Regression edged out Naive Bayes on overall accuracy and did noticeably better at actually catching positive tweets (recall of 0.53 vs 0.40). Both models are very good at spotting negative tweets (recall near 1.0 for the negative class), which makes sense given the negative tweets outnumber positive ones roughly 4 to 1 in this dataset - the class imbalance makes the negative class easier to nail and the positive class the harder one to get right.

## Files

- `NLP.ipynb` - the notebook
- `Tweets.csv` - dataset used

## Running it

```
pip install pandas numpy matplotlib seaborn scikit-learn wordcloud nltk
```

First run will need to download NLTK stopwords, the notebook handles that with `nltk.download('stopwords')`. `Tweets.csv` is already included in this repo, so it should work right out of the box.

## Notes

Given the class imbalance (roughly 4x more negative tweets than positive), accuracy alone is a bit misleading here - a model that just guessed "negative" every time would already score around 80%. That's why the classification report (precision/recall per class) tells a more honest story than the single accuracy number. Balancing the classes (oversampling positive tweets, or using class weights) would likely be the next thing to try to improve positive-class recall specifically.
