# Fake news Detection

# Importing required library
import streamlit as st
import joblib


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

 # #### Creating a function to convert the text in lowercase, remove the extra space, special chr., ulr and links.

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()


from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(random_state=0)


def Train_model(methodused):
        # ### Inserting fake and real dataset

    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")



    # Inserting a column called "class" for fake and real news dataset to categories fake and true news. 

    df_fake["class"] = 0
    df_true["class"] = 1



    # Removing last 10 rows from both the dataset, for manual testing  

    df_fake_manual_testing = df_fake.tail(10)
    for i in range(23480,23470,-1):
        df_fake.drop([i], axis = 0, inplace = True)
    df_true_manual_testing = df_true.tail(10)
    for i in range(21416,21406,-1):
        df_true.drop([i], axis = 0, inplace = True)



    # Merging the manual testing dataframe in single dataset and save it in a csv file

    df_fake_manual_testing["class"] = 0
    df_true_manual_testing["class"] = 1




    df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
    df_manual_testing.to_csv("manual_testing.csv")


    # Merging the main fake and true dataframe

    df_marge = pd.concat([df_fake, df_true], axis =0 )



    # #### "title",  "subject" and "date" columns is not required for detecting the fake news, so I am going to drop the columns.

    df = df_marge.drop(["title", "subject","date"], axis = 1)



    # #### Randomly shuffling the dataframe 

    df = df.sample(frac = 1)


    df.reset_index(inplace = True)
    df.drop(["index"], axis = 1, inplace = True)


    df["text"] = df["text"].apply(wordopt)


    # #### Defining dependent and independent variable as x and y

    x = df["text"]
    y = df["class"]


    # #### Splitting the dataset into training set and testing set. 

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


    # #### Convert text to vectors

    
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)


    # ### 1. Logistic Regression

    # from sklearn.linear_model import LogisticRegression

    # LR = LogisticRegression()
    if methodused == 1 :
        LR.fit(xv_train,y_train)
        st.write("Accuracy with Logistic Regression : ",LR.score(xv_test, y_test)*100,"%")
    
    print("hi1")
   

    # ### 2. Decision Tree Classification

    # from sklearn.tree import DecisionTreeClassifier

    # DT = DecisionTreeClassifier()
    if methodused == 2 :
        DT.fit(xv_train, y_train)
        st.write("Accuracy with Decision Tree Classifier : ",DT.score(xv_test, y_test)*100,"%")
    
    print("hi2")
 


    # ### 3. Random Forest Classifier

    # from sklearn.ensemble import RandomForestClassifier

    # RFC = RandomForestClassifier(random_state=0)
    if methodused == 3 :
        RFC.fit(xv_train, y_train)
        st.write("Accuracy with Random Forest Classifier : ",RFC.score(xv_test, y_test)*100,"%")


# # Model Testing With Manual Entry


def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Real News"
    

st.write("<h1 style='font-size: 48px;'>Fake News Detection</h1>", unsafe_allow_html=True)

st.write("<h3> By : Arpit Sahu (20U03003) & N RamaKrishna (20U03006)</h3>", unsafe_allow_html=True)

st.write("")    
st.write("")    
user_input = st.text_area("Paste The News Article", height=500)
st.write("")    
st.write("") 

st.session_state.radio_button = None

radio_button = st.radio(
    "Choose a Classifier Method ",
    ("Logistic Regression", "Decision Tree Classification", "Random Forest Classifier")
)
if radio_button != st.session_state.radio_button:
    st.session_state.radio_button = radio_button
st.write("")    
st.write("")  

def manual_testing_using_RFC(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_RFC = RFC.predict(new_xv_test)
    # Accuracy_RFC=RFC.score(xv_test, y_test)
    
    return output_lable(pred_RFC[0])

def manual_testing_using_LR(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    # Accuracy_LR=LR.score(xv_test, y_test)

    return output_lable(pred_LR[0])

def manual_testing_using_DT(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_DT = DT.predict(new_xv_test)
    # Accuracy_DT=DT.score(xv_test, y_test)
    
    return output_lable(pred_DT[0])



if st.button("Predict"):
    
    result = "def"
    if st.session_state.radio_button == "Logistic Regression":
        st.write("<h2 style='font-size: 30px;'>Prediction Using Logistic Regression</h2>", unsafe_allow_html=True)
        Train_model(1)
        result = manual_testing_using_LR(user_input)
        
    elif st.session_state.radio_button == "Decision Tree Classification":
        st.write("<h2 style='font-size: 30px;'>Prediction Using Decision Tree Classification</h2>", unsafe_allow_html=True)
        Train_model(2)
        result = manual_testing_using_DT(user_input);
        
    else:
        st.write("<h2 style='font-size: 30px;'>Prediction Using Random Forest Classifier</h2>", unsafe_allow_html=True)
        Train_model(3)
        result = manual_testing_using_RFC(user_input)
    
    if result == "Real News" :
        st.write("<h2 style='font-size: 30px; color: green'>Real News</h2>",unsafe_allow_html=True)
    else:
        st.write("<h2 style='font-size: 30px; color: red'>Fake News</h2>",unsafe_allow_html=True)
    
   

