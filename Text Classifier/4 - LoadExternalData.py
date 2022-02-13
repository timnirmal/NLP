import pickle
import random
import time
from statistics import mode

import nltk
from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
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


# time counter for program
start_time = time.time()

short_pos = open("positive.txt", "r").read()
short_neg = open("negative.txt", "r").read()

documents = []

for r in short_pos.split('\n'):
    documents.append((r, "pos"))

for r in short_neg.split('\n'):
    documents.append((r, "neg"))

save_documents = open("pickled_algos/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()


all_words = []

short_pos_words = nltk.word_tokenize(short_pos)
short_neg_words = nltk.word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

# Converting the words to frequency distribution
all_words_FrDis = nltk.FreqDist(all_words)
print("\n\n", all_words_FrDis.most_common(15))

word_features = list(all_words_FrDis.keys())[:5000]


save_word_features = open("pickled_algos/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


featureset = [(find_features(review), category) for (review, category) in documents]

random.shuffle(featureset)

# positive data example:
training_set = featureset[:10000]
testing_set = featureset[10000:]

classifier = nltk.NaiveBayesClassifier.train(training_set)  # NLTK Naive Bayes Classifier Training
print("Original Naive Bayes Classifier Algo Accuracy Percentage : ",
      (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15)

# Save the classifier
save_classifier = open("pickled_algos/originalnaivebayes5k.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()



# MultinomialNB Classifier
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)  # Scikit-Learn MultinomialNB Classifier Training
print("MNB_classifier Accuracy Percentage : ", (nltk.classify.accuracy(classifier, testing_set)) * 100)

# Save the MultinomialNB Classifier
save_classifier = open("pickled_algos/MNB_classifier5k.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()


# BernoulliNB Classifier
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)  # Scikit-Learn BernoulliNB Classifier Training
print("BernoulliNB_classifier Accuracy Percentage : ",
      (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)

# GaussianNB Classifier gives error because wrong data shape
# TypeError: A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.

# Save the BernoulliNB Classifier
save_classifier = open("pickled_algos/BernoulliNB_classifier5k.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()


# LogisticRegression Classifier
LogisticRegression_classifier = SklearnClassifier(LogisticRegression(max_iter=10000))
LogisticRegression_classifier.train(training_set)  # Scikit-Learn LogisticRegression Classifier Training
print("LogisticRegression_classifier Accuracy Percentage : ",
      (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

# Save the LogisticRegression Classifier
save_classifier = open("pickled_algos/LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()


# SGDClassifier Classifier
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)  # Scikit-Learn SGDClassifier Classifier Training
print("SGDClassifier_classifier Accuracy Percentage : ",
      (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)) * 100)

# Save the SGDClassifier Classifier
save_classifier = open("pickled_algos/SGDClassifier_classifier5k.pickle","wb")
pickle.dump(SGDClassifier_classifier, save_classifier)
save_classifier.close()


# SVC Classifier
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)  # Scikit-Learn SVC Classifier Training
print("SVC_classifier Accuracy Percentage : ", (nltk.classify.accuracy(SVC_classifier, testing_set)) * 100)

# Save the SVC Classifier
save_classifier = open("pickled_algos/SVC_classifier5k.pickle","wb")
pickle.dump(SVC_classifier, save_classifier)
save_classifier.close()

# LinearSVC Classifier
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)  # Scikit-Learn LinearSVC Classifier Training
print("LinearSVC_classifier Accuracy Percentage : ",
      (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)


# save the LinearSVC Classifier
save_classifier = open("pickled_algos/LinearSVC_classifier5k.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()


# NuSVC Classifier
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)  # Scikit-Learn NuSVC Classifier Training
print("NuSVC_classifier Accuracy Percentage : ", (nltk.classify.accuracy(NuSVC_classifier, testing_set)) * 100)

# Save the NuSVC Classifier
save_classifier = open("pickled_algos/NuSVC_classifier5k.pickle","wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()



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

end_time = time.time()
# time in minutes and seconds
print("\nTime taken: ", (end_time - start_time) / 60, " minutes")
