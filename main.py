import nltk
from nltk import word_tokenize, sent_tokenize

# For one time downloading, (after that comment out)
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('state_union')



# tokenizing means breaking a sentence into words
# lexicon means the words and their meanings
# corpora are sets of texts, usually a body of text
# ex: same word with different meaning (Based on the context, person who says, tone of voice, etc)

# grammer is how the words are combined to make sentences
# taggers are used to assign parts of speech to words
# chunkers are used to group words into phrases
# syntax parser is used to find the structure of sentences
# named entity recognizer finds the names of people, places, and things
# sentiment analyzer is used to find the sentiment of words
# wordnet is a large lexical database for the English language
# stopwords are common words that are usually not useful in a search
# stemmers are used to find the root word of a word
# lemmatizers are used to find the base word of a word
# q-grams are used to find similar words
# trigrams are used to find similar words
# n-grams are used to find similar words
# document clustering is used to group documents into groups
# document retrieval is used to find documents that are relevant to a query
# document summarization is used to find the most important sentences in a document
# information retrieval system is used to find relevant documents
# information retrieval is used to find relevant documents

example_text = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is " \
               "pinkish-blue. You shouldn't eat cardboard. "

print(example_text)

print("\nSentence Tokenizer:")
print(sent_tokenize(example_text))

for i in (sent_tokenize(example_text)):
    print(i)

print("\nWord tokenize:")
print(word_tokenize(example_text))

for i in (word_tokenize(example_text)):
    print(i)

# 2. Stop Words

from nltk.corpus import stopwords


example_sentence = "This is an example showing off stop word filtration."
stop_words = set(stopwords.words("english"))

words = word_tokenize(example_sentence)

print("\nStop words:")
print(stop_words)

print("\nFiltered sentence:")
filtered_sentence = [w for w in words if not w in stop_words]
# [for w in words, if not w in stop_words -> [TAB] filtered_sentences.append(w)]
print(filtered_sentence)

# 3. Stemming
# stemming is the process of reducing words to their stems
# stem is the root of a word
# for example:
#   - running -> run

from nltk.stem import PorterStemmer

ps = PorterStemmer()

example_words = ["python", "pythoner", "pythoning", "pythoned", "pythonly"]

new_text = "It is important to be pythonly while you are pythoning with python. All pythoners have pythoned poorly at " \
           "least once. "

print("\nStemmed text:")
for w in example_words:
    print(ps.stem(w))

for t in word_tokenize(new_text):
    print(ps.stem(t))


# 4. Lemmatization
# lemmatization is the process of finding the base word of a word


from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()


print("\nLemmatized text:")
for t in word_tokenize(new_text):
    print(lemmatizer.lemmatize(t))

# 5. Part of Speech Tagging
# part of speech tagging is the process of assigning parts of speech to words

print("\nPart of speech tagging:")
tagged_words = nltk.pos_tag(filtered_sentence)
print(tagged_words)

"""
pos tag list:

    CC	coordinating conjunction
    CD	cardinal digit
    DT	determiner
    EX	existential there (like: "there is" ... think of it like "there exists")
    FW	foreign word
    IN	preposition/subordinating conjunction
    JJ	adjective	            'big'
    JJR	adjective, comparative	'bigger'
    JJS	adjective, superlative	'biggest'
    LS  list maker 1)
    MD	modal   could, will
    NN	noun, singular          'desk'
    NNS	noun plural	            'desks'
    NNP	proper noun, singular	'Harrison'
    NNPS	proper noun, plural	'Americans'
    PDT	predeterminer	    'all the kids'
    POS	possessive ending	parent's
    PRP	personal pronoun	I, he, she
    PRP$    possessive pronuous	    my, his, hers
    RB	adverb	    very, silently,
    RBR	adverb,     comparative	better
    RBS	adverb,     superlative	best
    RP	particle	give up
    TO	to	go 'to' the store.
    UH	interjection	errrrrrrrm
    VB	verb, base form	take
    VBD	verb, past tense	    took
    VBG	verb, gerund/present    participle	taking
    VBN	verb, past participle	taken
    VBP verb, sing. present, non-3d     takes
    VBZ verb, 3rd person sing. present	takes
    WDT	wh-determiner	which
    WP	wh-pronoun	who, what
    WP$ possessive wh-pronoun	whose
    WRB	wh-adverb	where, when
"""

# With PunkSentenceTokenizer (Can be trained using dataset and then used for other purposes)

from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
# custom_sent_tokenizer is trained using the training text

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)
    except Exception as e:
        print(str(e))

process_content()



# 6. Chunking