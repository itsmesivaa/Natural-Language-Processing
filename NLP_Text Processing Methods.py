#Importing Necessary libraries for Text processing.
import nltk
nltk.download()
nltk.download("punkt")

#Sample Paragraph text to process
paragraph = """India holds a unique place on the global stage, marked by its profound cultural diversity, ancient heritage, 
               and significant economic impact. Known as the world's largest democracy, India's political system 
               exemplifies a robust engagement with democratic values, where millions participate in the electoral process, 
               showcasing the strength and resilience of its democratic institutions. Culturally, India is a mosaic of various 
               traditions, languages, religions, and festivals, each contributing to the rich tapestry that defines the nation. 
               Its spiritual depth is globally recognized, with major religions like Hinduism, Buddhism, Jainism, and 
               Sikhism originating here, offering a profound philosophical impact worldwide.Historically, 
               India has been a cradle of civilization for over five millennia, with a rich legacy of contributions to 
               science, mathematics, astronomy, and literature. The decimal and zero systems were developed by 
               Indian mathematicians, laying foundational stones for modern scientific and technological advancements.
               In terms of natural beauty, India's landscape is stunningly diverse, ranging from the snow-capped Himalayas in 
               the north to the sun-drenched beaches of the southern coast, from the arid deserts of the west to the 
               lush green forests of the east. This geographical diversity also fosters a remarkable variety of wildlife and 
               ecosystems, making India one of the 17 mega diverse countries in the world.Economically, 
               India is a powerhouse, noted for its rapid growth and extensive labor force. 
               It's a key player in global markets, particularly in IT, pharmaceuticals, and space technology, with ISRO's 
               achievements in satellite technology and missions to the Moon and Mars capturing the world’s imagination.
               Moreover, India’s culinary diversity is celebrated globally. Each region offers a unique blend of spices, 
               ingredients, and cooking techniques, making Indian cuisine a profound part of its cultural expression.
               Socially and politically, India plays a crucial role in international affairs, maintaining a strategic 
               position in South Asia and being a founding member of several international organizations like the 
               Non-Aligned Movement and the United Nations.All these elements combined make India a special and 
               integral part of the global community, celebrated for its contributions to culture, science, and 
               human development."""
               
print(paragraph)               

"""Tokenization method  a process in Natural Language Processing (NLP) that breaks down a sequence of text into smaller units called tokens. 
These tokens can be characters, words, phrases, sentences, or even dates and punctuation marks. 
"""
#Tokenizing above paragraph to sentences
tkn = nltk.sent_tokenize(paragraph)
print(tkn)

#Tokenizing above paragraph to individual words
words = nltk.word_tokenize(paragraph)
print(words)

#Stemming Process
"""
Stemming is a text preprocessing technique in natural language processing (NLP) that converts raw text data into a readable format. 
Stemming involves removing or modifying word endings or other affixes to reduce the inflected form of a word. For example, "am", "are", and "is" can be stemmed to "be". 
This process allows word forms that differ in non-relevant ways to be merged and treated as equivalent.

"""

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

#Creating Stemmer object for processing
stemming = PorterStemmer()

#Tokenizing raw paragraph text to sentences
stm_tkn_sent = nltk.sent_tokenize(paragraph)


for i in range(len(stm_tkn_sent)):
    #Tokenizing sentences to words during iteration for stemming process
    stm_wrd_tkn = nltk.word_tokenize(stm_tkn_sent[i])
    
    #Applying stemming process by elimination stopwords to tokenized words
    stm_wrd_tkn = [stemming.stem(x) for x in stm_wrd_tkn if x not in set(stopwords.words('english'))]
    
    #Again after stemming process forming the token sentences with pre-processed token words 
    stm_tkn_sent[i] = ' '.join(stm_wrd_tkn)
print(stm_tkn_sent)


#Lemmitization Process
"""
Lemmatization is a text pre-processing technique in natural language processing (NLP) that breaks down a word to its root meaning, also known as alemma. 
For example, the word "better" would be reduced to its root word, "lemme", or "good". 
Lemmatization is more accurate than stemming, but it's also more time consuming because it involves deriving the meaning of a word from something like a dictionary. 

"""


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#Creating Lemmatizer object for text processing
wnl = WordNetLemmatizer()

#Tokenizing raw paragraph text to sentences
lmt_tkn_sent = nltk.sent_tokenize(paragraph)


for i in range(len(lmt_tkn_sent)):
    #Tokenizing Paragraph sentences to words for futher lemmatization process
    lmt_wrd_tkn = nltk.word_tokenize(lmt_tkn_sent[i])
    
    #Applying lemmatization process by eliminating stopwords on tokenized words
    lmt_wrd_tkn = [wnl.lemmatize(x) for x in lmt_wrd_tkn if x not in set(stopwords.words('english'))]
        
    #Again after lemmatization process forming token words as sentences 
    lmt_tkn_sent[i] = " ".join(lmt_wrd_tkn)
    
print(lmt_tkn_sent)


#Bag of Words
"""Bag of words (BoW) is a Natural Language Processing (NLP) strategy that converts text into numbers based on word frequency, 
without considering the order or context of the words. The BoW model is a simple and intuitive approach to representing text data. 
It's used to preprocess text because algorithms in NLP work on numbers."""

print(paragraph)

#Cleaning the Data / Pre-processing before applying Bag of words vectorization

import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#Object Creation for Stemming & Lemmatization

stmg = PorterStemmer()
lmtz = WordNetLemmatizer()

#Tokenizing paragraph to sentences for further processing
para_tkn = nltk.sent_tokenize(paragraph)

#Stemming function to process text
def stemming(para_tkn):
    stm_txt = []
    for z in range(len(para_tkn)):
        txt = re.sub('^A-Za-z',' ',para_tkn[z])
        txt = re.sub(r"\'s", '',para_tkn[z]) #Removing 's from the text for better content
        txt = txt.lower().split()
        txt = [stmg.stem(x) for x in txt if x not in set(stopwords.words('english'))] # Stemming Process
        txt = ' '.join(txt)
        stm_txt.append(txt)
    return stm_txt

#Lemmatization function to process text
def lemmatization(para_tkn):
    lmtz_txt = []
    for z in range(len(para_tkn)):
        txt = re.sub('^A-Za-z',' ',para_tkn[z])
        txt = re.sub(r"\'s", '',para_tkn[z]) #Removing 's from the text for better content
        txt = txt.lower().split()
        txt = [lmtz.lemmatize(x) for x in txt if x not in set(stopwords.words('english'))] # Stemming Process
        txt = ' '.join(txt)
        lmtz_txt.append(txt)
    return lmtz_txt

#Calling Stemming function
final_stem_op = stemming(para_tkn)
print(final_stem_op)

#Calling lemmatization function
final_lemmatize_op = lemmatization(para_tkn)
print(final_lemmatize_op)


#Creating Bag of Words Model(Document Matrix) after pre-processing
#Nothing but storing features as vectors representation for the sentences and its word occurences.
from sklearn.feature_extraction.text import CountVectorizer

#Initializing object 
cv = CountVectorizer()

s_x = cv.fit_transform(final_stem_op).toarray()

l_x = cv.fit_transform(final_lemmatize_op).toarray()


#TF-IDF 
'''TF-IDF stands for term frequency-inverse document frequency, and it's a statistical method used in 
information retrieval (IR) and natural language processing (NLP) to measure the importance of a word 
to a document in a collection. It's also used in machine learning. TF-IDF is the product of two statistics: 
term frequency (TF) and inverse document frequency (IDF). TF is the frequency of a word, 
while IDF is a measure of how rare the word is across the corpus. TF-IDF adjusts for the fact that some words appear more frequently in general.'''

import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#Tokenizing paragraph to sentences for further processing
para_tkn = nltk.sent_tokenize(paragraph)

#Calling Lemmatization function to eradicate stopwords and to 
lmtz = WordNetLemmatizer()
final_lemmatize_op = lemmatization(para_tkn)
print(final_lemmatize_op)

#Creating TF-IDF models
from sklearn.feature_extraction.text import TfidfVectorizer

#Creating object for TF-IDF models
tf_idf = TfidfVectorizer()

tf_idfX = tf_idf.fit_transform(final_lemmatize_op).toarray()