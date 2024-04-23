#Fake News Classifier

import pandas as pd
import numpy as np
from tqdm import tqdm

#Reading CSV file 
fn_data = pd.read_csv(r"C:\Users\Admin\Datasets\Fake News Classifier Dataset\train.csv")

fn_data.head(40)

fn_data.columns

#Data Processing 
#Dropping null values from data
train_X = fn_data.dropna(axis=0,subset=['id', 'title', 'author', 'text', 'label']).reset_index()

#Creating training data by dropping label column which we going to predict
train_X = train_X.drop('label',axis=1)

fn_data.head(40)
train_X.head(40)

#Copying label column data 
y = fn_data['label']

y

fn_data
train_X.head(35)

#Copying base data to dataframe for processing lemmatization and Bagofwords
msg = train_X.copy()
#Dropping Index column 
msg = msg.drop(columns=['index'])
msg

#Data Pre-processing using NLP methods

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
#For Stemming 
from nltk.stem.porter import PorterStemmer
stm = PorterStemmer()

#For Lemmatization
from nltk.stem.wordnet import WordNetLemmatizer
lm = WordNetLemmatizer() #Object creation
import re
import tqdm

#Formatting text column by applying REGEX and Lemmatization 
def rephrase(b):
    final_lem_words = []
    for x in range(0,len(b)):
        txt = re.sub(r'[^A-Za-z]',' ', b[x])
        txt = re.sub(r"\'s", '',txt) #Removing 's from the text for better content
        txt = txt.lower().split()
        txt = [lm.lemmatize(word) for word in txt if not word in set(stopwords.words('english'))]
        txt = ' '.join(txt)
        final_lem_words.append(txt)        
    return final_lem_words

#Calling lemmatization function

x = rephrase(msg['text'])
print(x)
x[49]

#Creating Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500,ngram_range=(1,1))
X = cv.fit_transform(x).toarray()

#Listing sample set of features to see the combinations
cv.get_feature_names_out()[:200]

X.shape
y.shape
y[0:5]

#Dividing Dataset split into Train and Test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state= 1,shuffle= False)
print( y_train, y_test)

df = pd.DataFrame(X_train,columns=cv.get_feature_names_out())

df
msg['text'][0:6]

x[8]

import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
#Multinomial Algorithm    
    
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()

from sklearn import metrics
import numpy as np
import itertools

classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
score