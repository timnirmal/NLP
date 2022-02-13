import nltk
from nltk.corpus import stopwords

# nltk.download('stopwords')

# first 10 stopwords in english corpus
stopwords = stopwords.words('english')[:10]
print("stopwords : ", stopwords)

test_sentence = "This is my first test 12 string, Wow! we are learning nltk."

# remove punctuations
"""
import re
test_sentence = re.sub(r'[^\w\s]','',test_sentence)
print(test_sentence)
"""
no_punctuation = [word for word in test_sentence.split() if word.isalpha()]
print("no_punctuation : ", no_punctuation)

# remove stopwords
no_stopwords = [word for word in no_punctuation if word.lower() not in stopwords]
print("no_stopwords : ", no_stopwords)

# stemming
from nltk.stem import PorterStemmer
ps = PorterStemmer()
stemmed_words = [ps.stem(word) for word in no_stopwords]
print("stemmed_words : ", stemmed_words)


# lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in no_stopwords]
print("lemmatized_words : ", lemmatized_words)


# part of speech tagging
from nltk.tag import pos_tag
tagged_words = pos_tag(lemmatized_words)
print("tagged_words : ", tagged_words)


# named entity recognition
from nltk.chunk import ne_chunk
named_entities = ne_chunk(tagged_words)
print("named_entities : ", named_entities)
# draw tree
named_entities.draw()
