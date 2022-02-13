import nltk
from nltk import word_tokenize, sent_tokenize

# TODO: For one time downloading, (after that comment out)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('state_union')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')


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

# 1. Tokenization

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
# Chunking means
# For example: if we have a sentence like "I am a
# student. I am studying in the university."
# We can chunk the sentence into different parts like
# I am a student
# I am studying in the university
# So Most of the time we use the chunking to find the subject of the sentence
# Which means this happens around a Noun

# Before chunking you need to know Regular Expressions
"""


Identifiers:

\d = any number
\D = anything but a number
\s = space
\S = anything but a space
\w = any letter
\W = anything but a letter
. = any character, except for a new line
\b = space around whole words
\. = period. must use backslash, because . normally means any character.

Modifiers:
    {1,3} = for digit, you expect 1-3 count of digital numbers or "places"
    + = match 1 or more
    ? = match 0 or 1 repetitions
    * = match 0 or more repetitions
    $ = match the end of the string
    ^ = match the beginning of the string
    | = match either of the patterns
    [] = range or variance, [a-z] = match characters between a to z (both included)
    {x} = expect to see this amount of the preceding code  / exact number of repetitions
    {x,y} = expect to see this x to y amount of the preceding code 
    
White Space Charts:

\n = new line
\s = space
\t = tab
\e = escape
\f = form feed
\r = carriage return
Characters to REMEMBER TO ESCAPE IF USED!

. + * ? [ ] $ ^ ( ) { } | \
Brackets:

[] = quant[ia]tative = will find either quantitative, or quantatative.
[a-z] = return any lowercase letter a-z
[1-5a-qA-Z] = return all numbers 1-5, lowercase letters a-q and uppercase A-Z

"""

"""
Chunking is the process of grouping words together to form a phrase
(Part of Speech Tagging + Regex Chunking = Chunking) 

POS Tagging: is denoted by < and >
RegEx symbols are denote with them.
"""


def process_content_with_Chunking():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            chunkGram = r""""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""

            chunkPharser = nltk.RegexpParser(chunkGram)
            chunked = chunkPharser.parse(tagged)

            print(chunked)
            print("\n")
            # chunked.draw()

            """
            # Chuned and Non Chunked subtree
            for subtree in chunked.subtrees():
                print("\nSubtree : ", subtree)

            # Chunked and Non Chunked leaves
            for leaf in chunked.leaves():
                print("Leaf : ", leaf)

            # Chunked subtree
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                print("\nSubtree Chunked: ", subtree)

                # Chunked leaves
                for leaf in subtree.leaves():
                    print("Leaf Chunked: ", leaf)
            """

    except Exception as e:
        print(str(e))


# Uncomment
print("\n\n\n Chunking : ")
process_content_with_Chunking()


# 7. Chinking
# Chinking means to remove the chunk from the sentence

def process_content_with_Chinking():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            chunkGram = r""""Chunk: {<.*>?}
                                    }<VB.?|DT|DT|TO>{"""  # Note that we use }{ instead of {} for chinking

            chunkPharser = nltk.RegexpParser(chunkGram)
            chunked = chunkPharser.parse(tagged)

            print(chunked)
            print("\n")
            # chunked.draw()

    except Exception as e:
        print(str(e))


# Uncomment
# process_content_with_Chinking()

# 8. Named Entity Recognition
# Named Entity Recognition is the process of finding the name of the person, place, organization, etc.
# In NLP, we use the chunking to find the subject of the sentence
# But in NER, we use the chunking to find the name of the person, place, organization, etc.

"""
Name Entity types:
    ORGANIZATION - Georgia-Pacific Corp., WHO
    PERSON - Eddy Bonte, President Obama
    LOCATION - Murray River, Mount Everest
    DATE - June, 2008-06-29
    TIME - two fifty a m, 1:30 p.m.
    MONEY - 175 million Canadian Dollors, GBP 10.40
    PERCENT - twenty pct, 18.75 %
    FACILITY - Washington Monument, Stonehenge
    GPE - South East Asia, Midlothian
"""


def process_content_with_NameEntity():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            nameEnt = nltk.ne_chunk(tagged, binary=True)
            # binary = False means that we are looking for the subject of the sentence
            # ex: Chunk will show as PERSON and its branch will show as the name of the person chunked (George Bush)

            # binary = True means that we are looking for the name of the person, place, organization, etc.
            # ex: Chunk as EN (Entity)

            print(nameEnt)
            print("\n")
            nameEnt.draw()

    except Exception as e:
        print(str(e))


# Uncomment
# process_content_with_NameEntity()

# 9. Lemmatization

# Lemmatization is the process of finding the root word of the word like "is" to "be" or "play" to "play" etc.(Not Sure)
# Similar to stemming, but it is more accurate and less prone to errors (Which means same word or a synonym)

# Default pos = Noun

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print("\nLemmatization")
print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"), end="\n\n")

# with pos
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("run", pos="v"))

# Corpora Path - C:\Users\timni\PycharmProjects\NLP\venv\Lib\site-packages\nltk

# 10. WordNet
"""WordNet is the lexical database i.e. dictionary for the English language, specifically designed for natural 
language processing. 

Synset is a special kind of a simple interface that is present in NLTK to look up words in WordNet. Synset instances 
are the groupings of synonymous words that express the same concept. Some of the words have only one Synset and some 
have several. """

from nltk.corpus import wordnet

synset = wordnet.synsets("program")

print("\nWordNet")
print(synset[0])
print(synset[0].name())  # sysnet object
print(synset[0].lemmas())  # list of lemmas
# lemmas() returns a list of lemmas, each of which has a name, part of speech, and an example sentence
# lemmas means the base word of the word something like that
for i in synset[0].lemmas():
    print(i)
    print(i.name())
    print(i.synset())
print()

print(synset[0].lemmas()[0].name())  # sysnet name
print(synset[0].definition())
print(synset[0].examples())

# Example code : Synonyms and Antonyms

synonyms = []
antonyms = []

for synset in wordnet.synsets("good"):
    for lemma in synset.lemmas():
        print("l:", lemma)
        synonyms.append(lemma.name())
        if lemma.antonyms():
            antonyms.append(lemma.antonyms()[0].name())

print("\nSynonyms and Antonyms")
print(set(synonyms))
print(set(antonyms))

# Similarity between two words
#   Check on WUP_similarity

print("\nSimilarity between two words")

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")
print(w1.wup_similarity(w2))

w3 = wordnet.synset("car.n.01")
print(w1.wup_similarity(w3))

w4 = wordnet.synset("cat.n.01")
print(w1.wup_similarity(w4))

w5 = wordnet.synset("cactus.n.01")
print(w1.wup_similarity(w5))
