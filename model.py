import pandas as pd
import pickle


dataset=pd.read_csv('spamham.csv')
dataset=dataset.iloc[:,:2]
dataset.rename(columns={'v1':'labels','v2':'messages'},inplace=True)


#-------------------------------------------------------------#
### BALANCING THE DATASET

spam=dataset[dataset.labels=='spam']
ham=dataset[dataset.labels=='ham']

ham=ham.sample(spam.shape[0])

dataset=ham.append(spam,ignore_index=True)


#-------------------------------------------------------------#

# Importing RE to clean the data

import re 

# Replace email addresses with 'email'
processed = dataset['messages'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',
                                 'emailaddress')

# Replace URLs with 'webaddress'
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
                                  'webaddress')

# Replace money symbols with 'moneysymb' 
processed = processed.str.replace(r'£|\$', 'moneysymb')
    
# Replace 10 digit phone numbers (formats include paranthesis, spaces, 
#                                   no spaces, dashes) with 'phonenumber'
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                  'phonenumbr')
    
# Replace numbers with 'numbr'
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')

# Remove punctuation
processed = processed.str.replace(r'[^\w\d\s]', ' ')

# Replace whitespace between terms with a single space
processed = processed.str.replace(r'\s+', ' ')

# Remove leading and trailing whitespace
processed = processed.str.replace(r'^\s+|\s+?$', '')

processed = processed.str.lower()



#-------------------------------------------------------------#

## WORD CLOUD
from wordcloud import WordCloud

spam=dataset[dataset.labels=='spam']['messages']
ham=dataset[dataset.labels=='ham']['messages']


spam_words=[]
ham_words=[]

import nltk

spam_index=list(spam.index)
for i in spam_index:
    words=[word for word in nltk.word_tokenize(spam[i])]
    spam_words=spam_words+words

ham_index=list(ham.index)
for i in ham_index:
    words=[word for word in nltk.word_tokenize(ham[i])]
    ham_words=ham_words+words
    
    
import matplotlib.pyplot as plt    
spam_wordcloud=WordCloud(width=600,height=400).generate(" ".join(spam_words))
plt.figure(figsize=(10,8),facecolor='k')
plt.imshow(spam_wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

ham_wordcloud=WordCloud(width=600,height=400).generate(" ".join(ham_words))
plt.figure(figsize=(10,8),facecolor='k')
plt.imshow(ham_wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


#-------------------------------------------------------------#

# Creating Corpus

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
ps=PorterStemmer()
from nltk.stem import WordNetLemmatizer
wl=WordNetLemmatizer()
corpus=[]

for i in range(len(processed)):
    review=processed[i].split()
    
    reviews=[wl.lemmatize(word) for word in review if not word in 
            stopwords.words('english')]
    review_final = ' '.join(reviews)
    corpus.append(review_final)


#-------------------------------------------------------------#


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)

X=cv.fit_transform(corpus).toarray()
pickle.dump(cv,open('change.pkl','wb'))

y=pd.get_dummies(dataset['labels'])
y=y.iloc[:,1].values

print(cv.get_feature_names())

#-------------------------------------------------------------#

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.20, 
                                                    random_state = 0,
                                                    shuffle=True)



#-------------------------------------------------------------#

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X,y)

pickle.dump(spam_detect_model, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

y_pred=spam_detect_model.predict(X_train)



#-------------------------------------------------------------#
from sklearn.metrics import confusion_matrix,accuracy_score
print(accuracy_score(y_train,y_pred))
cm=confusion_matrix(y_train,y_pred)
    