#SMS Spam Classification

import pandas as pd

#Reading Data from location 
file_det = pd.read_csv(r"C:\Users\Admin\Automation_Python\NLP Projects\spam.csv", encoding='ANSI', sep=',', usecols= ['v1','v2'] )

#Renaming column names to readable form
file_det.rename(columns={'v1':'label','v2':'content'},inplace= True)

file_det


import nltk
nltk.download()
nltk.download("stopwords")
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()

from nltk.stem import WordNetLemmatizer
wl = WordNetLemmatizer()

final_txt = []

#Text processing 

for x in range(0,len(file_det)):
    txt = re.sub('[^A-Za-z]',' ', file_det['content'][x])
    txt = txt.lower()
    txt = txt.split()
    txt = [wl.lemmatize(word) for word in txt if not word in set(stopwords.words('english'))]
    txt = ' '.join(txt)
    final_txt.append(txt)

final_txt

#Creating Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)

cv.fit_transform(final_txt)

X = cv.fit_transform(final_txt).toarray()

#Converting Ham and Spam column to dummies into machine readable format

y = pd.get_dummies(file_det['label'])
#Picking only Spam column
y = y.iloc[:,1].values
y

#Train Test Split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.20, random_state=0)


#Training model using Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect = MultinomialNB().fit(X_train,y_train)

y_pred = spam_detect.predict(X_test)


from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test,y_pred)

#Checking accuracy

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)