# IMDB-Reviews-Sentiment-Anaylsis
Training a model to predict positive/negative sentiments from a movie's review on IMDB

## Bag of Words Implementation 

We first began by training a baseline model using the Bag of Words (BoG) algorithm, using guidance from the instructions provided <a href="https://www.kaggle.com/c/word2vec-nlp-tutorial/overview">here</a>.

To evaluate our model, we used the Keggle's submission option that allocated us a score based on the area under our model's ROC curve. Using the BoG implementation, we were able to obtain our default scores (listed below). For our classifier, we utilized Scikit-Learn's Random Forest Classifier, first with a default of ```n_estimators = 100 ``` and later (at the cost of greater time required to train the model) with ```n_estimators = 400```. While we noticed an increase in our score, it was quite minute compared to the increase in time.

| Implementation | Result |
| -------------- | ------ |
| BoG, 100 trees | 0.8432 |
| BoG, 400 trees | 0.8590 |
| BoG, 1000 trees| 0.8558 |

As we can clearly observe, increasing the number of trees in our RF classifier worked to a certain extent, yet after a cut-off point, it stopped increasing our score and was actually detrimental. 
