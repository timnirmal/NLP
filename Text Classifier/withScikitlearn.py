import random

import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import movie_reviews
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC, NuSVC

# Now let's get the code from Text-Classifier.py to compare that with the Scikit-Learn Classifiers

documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = [w.lower() for w in movie_reviews.words()]  # List of all words in movie_reviews in lower case

# Converting the words to frequency distribution
all_words_FrDis = nltk.FreqDist(all_words)
print("\n\n", all_words_FrDis.most_common(15))

word_features = list(all_words_FrDis.keys())[:3000]


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


featureset = [(find_features(review), category) for (review, category) in documents]
training_set = featureset[:1900]
testing_set = featureset[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)  # NLTK Naive Bayes Classifier Training
print("Original Naive Bayes Classifier Algo Accuracy Percentage : ",
      (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15)

# MultinomialNB Classifier
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)  # Scikit-Learn MultinomialNB Classifier Training
print("MNB_classifier Accuracy Percentage : ", (nltk.classify.accuracy(classifier, testing_set)) * 100)

# BernoulliNB Classifier
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)  # Scikit-Learn BernoulliNB Classifier Training
print("BernoulliNB_classifier Accuracy Percentage : ",
      (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)

# GaussianNB Classifier gives error because wrong data shape
# TypeError: A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.


# LogisticRegression Classifier
LogisticRegression_classifier = SklearnClassifier(LogisticRegression(max_iter=1000))
LogisticRegression_classifier.train(training_set)  # Scikit-Learn LogisticRegression Classifier Training
print("LogisticRegression_classifier Accuracy Percentage : ",
      (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

# SGDClassifier Classifier
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)  # Scikit-Learn SGDClassifier Classifier Training
print("SGDClassifier_classifier Accuracy Percentage : ",
      (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)) * 100)

# SVC Classifier
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)  # Scikit-Learn SVC Classifier Training
print("SVC_classifier Accuracy Percentage : ", (nltk.classify.accuracy(SVC_classifier, testing_set)) * 100)

# LinearSVC Classifier
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)  # Scikit-Learn LinearSVC Classifier Training
print("LinearSVC_classifier Accuracy Percentage : ",
      (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

# NuSVC Classifier
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)  # Scikit-Learn NuSVC Classifier Training
print("NuSVC_classifier Accuracy Percentage : ", (nltk.classify.accuracy(NuSVC_classifier, testing_set)) * 100)

"""
Original Naive Bayes Classifier Algo Accuracy Percentage :  83.0
MNB_classifier Accuracy Percentage :  83.0
BernoulliNB_classifier Accuracy Percentage :  83.0

LogisticRegression_classifier Accuracy Percentage :  82.0
(ConvergenceWarning: lbfgs failed to converge (status=1): STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.)
Solution found in:
https://stackoverflow.com/questions/62658215/convergencewarning-lbfgs-failed-to-converge-status-1-stop-total-no-of-iter
https://stackoverflow.com/questions/38640109/logistic-regression-python-solvers-definitions/52388406#52388406

SGDClassifier_classifier Accuracy Percentage :  80.0
SVC_classifier Accuracy Percentage :  84.0
LinearSVC_classifier Accuracy Percentage :  78.0
NuSVC_classifier Accuracy Percentage :  83.0
"""

"""
After max_iter=1000, in the LogisticRegression_classifier 

Original Naive Bayes Classifier Algo Accuracy Percentage :  82.0
MNB_classifier Accuracy Percentage :  82.0
BernoulliNB_classifier Accuracy Percentage :  81.0
LogisticRegression_classifier Accuracy Percentage :  85.0
SGDClassifier_classifier Accuracy Percentage :  84.0
SVC_classifier Accuracy Percentage :  82.0
LinearSVC_classifier Accuracy Percentage :  81.0
"""

"""
Original Naive Bayes Classifier Algo Accuracy Percentage :  79.0
Most Informative Features
                  annual = True              pos : neg    =      9.9 : 1.0
                   sucks = True              neg : pos    =      9.6 : 1.0
           unimaginative = True              neg : pos    =      8.1 : 1.0
                 frances = True              pos : neg    =      7.8 : 1.0
                 idiotic = True              neg : pos    =      7.3 : 1.0
                    mena = True              neg : pos    =      6.8 : 1.0
                  shoddy = True              neg : pos    =      6.8 : 1.0
             silverstone = True              neg : pos    =      6.8 : 1.0
                  suvari = True              neg : pos    =      6.8 : 1.0
                obstacle = True              pos : neg    =      6.5 : 1.0
               atrocious = True              neg : pos    =      6.4 : 1.0
              schumacher = True              neg : pos    =      6.4 : 1.0
                   groan = True              neg : pos    =      6.2 : 1.0
                 kidding = True              neg : pos    =      6.2 : 1.0
                  turkey = True              neg : pos    =      6.2 : 1.0
MNB_classifier Accuracy Percentage :  79.0
BernoulliNB_classifier Accuracy Percentage :  80.0
LogisticRegression_classifier Accuracy Percentage :  82.0
SGDClassifier_classifier Accuracy Percentage :  81.0
SVC_classifier Accuracy Percentage :  81.0
LinearSVC_classifier Accuracy Percentage :  80.0
NuSVC_classifier Accuracy Percentage :  83.0
"""
