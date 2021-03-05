
# # SVM classifier: Score is 0.834705
# # SVM classifier with bigrams: Score is 0.838555
# # SVM classifier with bigrams and eliminate stop words: Score is 0.84014

# script for 1a

# change classifier to SVM

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data) #list of strings
# print(tfidf_Vect.vocabulary_)
clf = SVC(kernel = 'linear') #changed this to SVM
clf.fit(X_train_tfidf, twenty_train.target) # list of numbers between 0 to 19 inclusive

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print(score)

# script for 1b

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

# include bigrams in our vocab

tfidf_Vect = TfidfVectorizer(ngram_range = (1,2)) # to include bigrams
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data) #list of strings
# print(tfidf_Vect.vocabulary_)
clf = SVC(kernel = 'linear') #changed this to SVM
clf.fit(X_train_tfidf, twenty_train.target) # list of numbers between 0 to 19 inclusive

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print(score)

# script for 1c

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

# eliminate stop words which helps make the matrix less sparse

tfidf_Vect = TfidfVectorizer(stop_words = 'english', ngram_range = (1,2)) # eliminate stop words
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data) #list of strings
# print(tfidf_Vect.vocabulary_)
clf = SVC(kernel = 'linear') #changed this to SVM
clf.fit(X_train_tfidf, twenty_train.target) # list of numbers between 0 to 19 inclusive

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print(score)

