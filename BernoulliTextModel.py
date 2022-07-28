import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

class model:
    def __init__(self):
        self.mod = BernoulliNB()
        self.X = np.array([])
        self.y = np.array([])
        self.clf = ExtraTreesClassifier(n_estimators=100)
        self.T1 = CountVectorizer(stop_words='english')
        self.T2 = CountVectorizer(token_pattern=r'\:\)\)|\:\-D|\:\-\(|\:\-\/|\:\'\(|\:\(\(|\:\-\)|\;\-\)|\:\)|\:/|\:\(|\:X|\:D|\:P|\:\||\;\)|\;\(|\;p|[\U00002600-\U000027BF]|[\U0001f300-\U0001f64F]|[\U0001f680-\U0001f6FF]')
        self.T3 = T3 = CountVectorizer(token_pattern=r'[\!\?\.\,]+')

    def __preprocessString(self, string):
        #remove websites
        string = re.sub(r'http[s]?:[^ ]*', '', string)

        #replace wacky characters
        string = re.sub('&amp;', '&', string)
        string = re.sub('&lt;', '<', string)
        string = re.sub('&gt;', '>', string)

        #search for emojis
        emojis = []
        emojis += re.findall(r'\:\)\)|\:\-D|\:\-\(|\:\-\/|\:\'\(|\:\(\(|\:\-\)|\;\-\)', string)
        string = re.sub(r'\:\)\)|\:\-D|\:\-\(|\:\-\/|\:\'\(|\:\(\(|\:\-\)|\;\-\)', '', string)
        emojis += re.findall(r'\:\)|\:/|\:\(|\:X|\:D|\:P|\:\||\;\)|\;\(|\;p', string)
        string = re.sub(r'\:\)|\:/|\:\(|\:X|\:D|\:P|\:\||\;\)|\;\(|\;p', '', string)
        emojis += re.findall(r'[\U00002600-\U000027BF]|[\U0001f300-\U0001f64F]|[\U0001f680-\U0001f6FF]', string)
        string = re.sub(r'[\U00002600-\U000027BF]|[\U0001f300-\U0001f64F]|[\U0001f680-\U0001f6FF]', '', string)

        #search for punctuation
        punctuation = []
        punctuation += re.findall(r'[\!\?\.\,]+', string)
        string = re.sub(r'[\!\?\.\,]', '', string)

        #remove special chars from the string
        string = re.sub(r'[\'\’]', '', string)
        string = re.sub(r'[^a-zA-Z0-9]+', ' ', string)
        string = re.sub(r'[\s]+', ' ', string)
        string = string.strip().lower()

        return [string, emojis, punctuation]
    
    def train(self, X, y):
        temp = [self.__preprocessString(x) for x in X]
        text = [x[0] for x in temp]
        emojis = [' '.join(x[1]) for x in temp]
        punct = [' '.join(x[2]) for x in temp]

        X1 = self.T1.fit_transform(text).toarray()
        self.clf.fit(X1, y)
        self.select = SelectFromModel(self.clf, prefit=True)
        X1 = self.select.transform(X1)
        X2 = self.T2.fit_transform(emojis).toarray()
        X3 = self.T3.fit_transform(punct).toarray()
        self.X = np.concatenate((X1, X2, X3), axis=1)
        self.y = y
        self.mod.fit(self.X, self.y)

    def predict(self, text):
        x = self.__preprocessString(text)
        x1 = self.T1.transform([x[0]]).toarray()
        x1 = self.select.transform(x1)
        x2 = self.T2.transform([' '.join(x[1])]).toarray()
        x3 = self.T3.transform([' '.join(x[2])]).toarray()
        x = np.append(np.append(x1, x2), x3)
        return self.mod.predict([x])[0]

    def testAccuracy(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, stratify=self.y, test_size=.2)
        testModel = BernoulliNB()
        testModel.fit(X_train, y_train)
        return testModel.score(X_test, y_test)