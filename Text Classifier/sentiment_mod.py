import pickle
import time
from statistics import mode

from nltk.classify import ClassifierI


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

# load pickled data from file word_features.pickle
word_features5k_f = open("pickled_algos/word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


# Load the original classifier
classifier_f = open("pickled_algos/originalnaivebayes5k.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

# Load the MultinomialNB Classifier
MNB_classifier_f = open("pickled_algos/MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(MNB_classifier_f)
MNB_classifier_f.close()

# Load the BernoulliNB Classifier
BernoulliNB_classifier_f = open("pickled_algos/BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(BernoulliNB_classifier_f)
BernoulliNB_classifier_f.close()

# Load the LogisticRegression Classifier
LogisticRegression_classifier_f = open("pickled_algos/LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(LogisticRegression_classifier_f)
LogisticRegression_classifier_f.close()

# Load SGDClassifier Classifier
SGDClassifier_classifier_f = open("pickled_algos/SGDClassifier_classifier5k.pickle", "rb")
SGDClassifier_classifier = pickle.load(SGDClassifier_classifier_f)
SGDClassifier_classifier_f.close()

# Load SVC Classifier
SVC_classifier_f = open("pickled_algos/SVC_classifier5k.pickle", "rb")
SVC_classifier = pickle.load(SVC_classifier_f)
SVC_classifier_f.close()

# Load Linear SVC Classifier
LinearSVC_classifier_f = open("pickled_algos/LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(LinearSVC_classifier_f)
LinearSVC_classifier_f.close()

# Load NuSVC Classifier
NuSVC_classifier_f = open("pickled_algos/NuSVC_classifier5k.pickle", "rb")
NuSVC_classifier = pickle.load(NuSVC_classifier_f)
NuSVC_classifier_f.close()

########################################################################################################################

# Voting Classifier with set of Classifiers
voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)


########################################################################################################################

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)


end_time = time.time()
print("\nTime taken: ", (end_time - start_time) / 60, " minutes\n")
