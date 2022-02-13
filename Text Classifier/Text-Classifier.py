import pickle
from typing import List, Tuple, Dict, Any

import nltk
import random
from nltk.corpus import movie_reviews

# nltk.download('movie_reviews')

# Data For training the classifier (Feature)
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

print("Document : ",documents[1])  # (['the', 'movie', 'is', 'not', 'great'], 'neg') -> List of words
print("\n\n  Pos, ", documents[1][0])  # List of words in Document[1]
print("\n\n NEg , ", documents[1][1])  # Category- positive or negative -> Category of Document[1]

all_words = [w.lower() for w in movie_reviews.words()]  # List of all words in movie_reviews in lower case
print(all_words[:10])

# Converting the words to frequency distribution
all_words_FrDis = nltk.FreqDist(all_words)
print("\n\n", all_words_FrDis.most_common(15))

print(all_words_FrDis["stupid"])  # Frequency of the word "stupid" -> 253

print("Keys")
word_features = list(all_words_FrDis.keys())[:3000]


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featureset: list[tuple[dict[Any, bool], Any]] = []
for (review, category) in documents:
    featureset.append((find_features(review), category))

print("Feature Set ", featureset[0]) # Check if each feature is in the document[:3000] or not and assign true, false

# From feature set lets make training set and testing set

training_set = featureset[:1900]
testing_set = featureset[1900:]

# Naive Bayes Classifier Algorithm
# This is simple algorithm and Can be scaled very much (TODO: Search on this)
# posterior = prior occurrences x likelihood / evidence

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes Classifier Algo Accuracy Percentage : ", (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15)

""""
Output will be something like this:

Naive Bayes Classifier Algo Accuracy Percentage :  79.0
Most Informative Features
                   sucks = True              neg : pos    =     10.1 : 1.0
                  annual = True              pos : neg    =      9.4 : 1.0
           unimaginative = True              neg : pos    =      8.5 : 1.0
                 frances = True              pos : neg    =      8.1 : 1.0
             silverstone = True              neg : pos    =      7.9 : 1.0
               atrocious = True              neg : pos    =      7.2 : 1.0
                 idiotic = True              neg : pos    =      7.2 : 1.0


"""


# Save Classifier
save_classifier = open("naivebayes.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

# Load Classifier
classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

print("Naive Bayes Classifier Algo Accuracy Percentage : ", (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15)
