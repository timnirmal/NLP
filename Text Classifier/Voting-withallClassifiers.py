import random
from statistics import mode

import nltk
from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import movie_reviews
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC, NuSVC


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


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
LogisticRegression_classifier = SklearnClassifier(LogisticRegression(max_iter=10000))
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

########################################################################################################################

# Voting Classifier with set of Classifiers
voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)
print("vote_classifier Accuracy Percentage : ", (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)

########################################################################################################################

# test with testdata

print("\nClassification: ", voted_classifier.classify(find_features(testing_set[0][0])), "Confidence %: ",
      voted_classifier.confidence(find_features(testing_set[0][0])) * 100)
print("Classification: ", voted_classifier.classify(find_features(testing_set[1][0])), "Confidence %: ",
      voted_classifier.confidence(find_features(testing_set[1][0])) * 100)
print("Classification: ", voted_classifier.classify(find_features(testing_set[2][0])), "Confidence %: ",
      voted_classifier.confidence(find_features(testing_set[2][0])) * 100)
print("Classification: ", voted_classifier.classify(find_features(testing_set[3][0])), "Confidence %: ",
      voted_classifier.confidence(find_features(testing_set[3][0])) * 100)
print("Classification: ", voted_classifier.classify(find_features(testing_set[4][0])), "Confidence %: ",
      voted_classifier.confidence(find_features(testing_set[4][0])) * 100)
print("Classification: ", voted_classifier.classify(find_features(testing_set[5][0])), "Confidence %: ",
      voted_classifier.confidence(find_features(testing_set[5][0])) * 100)


def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)


print(sentiment("I love this car"))
print(sentiment("This view is horrible"))
print(sentiment("I feel great this morning"))
print(sentiment("I am so excited about the concert"))
print(sentiment("I am not feeling well today"))
print(sentiment("He is not here today"))
print(sentiment("I have to go"))
print(sentiment("I am going to come"))
print(sentiment("I am coming"))
