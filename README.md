# IMDB-Reviews-Sentiment-Anaylsis
Training a model to predict positive/negative sentiments from a movie's review on IMDB

## Bag of Words Implementation 

I first began by training a baseline model using the Bag of Words (BoG) algorithm, using guidance from the instructions provided <a href="https://www.kaggle.com/c/word2vec-nlp-tutorial/overview">here</a>.

To evaluate my model, I used the Keggle's submission option that allocated a score based on the area under my model's ROC curve. Using the BoG implementation, I was able to obtain my default scores (listed below). For my classifier, I utilized Scikit-Learn's Random Forest Classifier, first with a default of ```n_estimators = 100 ``` and later (at the cost of greater time required to train the model) with ```n_estimators = 400```. While I noticed an increase in my score, it was quite minute compared to the increase in time.

| Implementation | Result |
| -------------- | ------ |
| BoG, 100 trees | 0.8432 |
| BoG, 400 trees | 0.8590 |
| BoG, 1000 trees| 0.8558 |

As we can clearly observe, increasing the number of trees in my RF classifier worked to a certain extent, yet after a cut-off point, it stopped increasing my score and was actually detrimental. 

## Word2Vec Average Vector Implementation

After establishing a baseline using the BoG model, I moved on to using the Word2Vec Neural Network algorithm established by Google, with documentation found <a href="https://radimrehurek.com/gensim/models/word2vec.html">here</a> and using guidance in implementing it from <a href="https://www.kaggle.com/c/word2vec-nlp-tutorial/overview">here</a>. 

I finetuned my hyper-parameters, as per the recommendations established in this research paper by Stanford found <a href="https://cs224d.stanford.edu/reports/SadeghianAmir.pdf">here</a>. Moreover, to train the model, I used the 75,000 reviews (25,000 labeled and 50,000 unlabeled) provided by Kaggle. 

However, the results of the implementation itself were disappointing since I failed to notice an improvement in score, rather suffering a slight decrease from my BoG implementation 

| Implementation | Result |
| -------------- | ------ |
| BoG, max score | 0.8590 |
| W2V, Avg Vec | 0.8314 |
