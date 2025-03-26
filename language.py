import pandas as pd
import numpy as np
import re 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
from sklearn.preprocessing import LabelEncoder

warnings.simplefilter("ignore")

df = pd.read_csv("Language Detection.csv")
print(df.head(10))
print(df['Language'].value_counts())


X = df['Text']
y = df['Language']

le = LabelEncoder()
y = le.fit_transform(y)
print(y)

#TEXT PREPROCESSING because this data set was scraped from Wikipedia, hence it contains symbols which may affect quality of model

#create a list for appending the preprocessed text 
data_list = []
for text in X:
	text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
	text = re.sub(r'[[]]', ' ', text)
	text = text.lower()
	data_list.append(text)

#Bag of words - converting text into numerical form by creating bag of words using count vectorizer 

from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer()
X = cv.fit_transform(data_list).toarray()
print(X.shape)

#Train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy is:",ac)

#---------------------------------Plot figure 

#plt.figure(figsize=(15,10))
#sns.heatmap(cm, annot=True)
#plt.show()


#-----------------------------#

def predict(text):
     x = cv.transform([text]).toarray() # converting text to bag of words model (Vector)
     lang = model.predict(x) # predicting the language
     lang = le.inverse_transform(lang) # finding the language corresponding the the predicted value
     print("The langauge is in",lang[0]) # printing the language

print(predict('I am a data scientist'))


