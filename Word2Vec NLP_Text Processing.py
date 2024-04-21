#Word2Vec NLP_Text Processing 

#Importing Necessary libraries for Text processing.
import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import re

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

#Text Data pre-processing
txt = re.sub(r'[^a-zA-Z]',' ',paragraph)
txt = re.sub(r'[0-9]','',paragraph)
txt = txt.lower()

print(txt)

#Tokenizing the pre-processed data into sentences

#Sentence Tokenizing
sentences = nltk.sent_tokenize(txt)
print(sentences)

#Word tokenizing
sentences = [nltk.word_tokenize(x) for x in sentences]
print(sentences)

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in set(stopwords.words('english'))]


print(sentences)

#Training the Word2Vec model
model = Word2Vec(sentences,min_count=1)

words = model.wv.key_to_index

model.wv.key_to_index['landscape']

similar = model.wv.most_similar('india')

similar