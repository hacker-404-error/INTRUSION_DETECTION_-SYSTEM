# Importing Libraries
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import time


# Set the title and page configuration
st.set_page_config(page_title='IDS with SVM', layout='wide')

# Add a title to the page
st.title('Intrusion Detection System with SVM')

# Add a file uploader to allow the user to select a CSV file
uploaded_file = st.file_uploader('Upload a CSV file')

# -------------------------------------------------------------------------------------------------------------------------
# Define a function to train and test the SVM model
def train_test_model(df):
    global Y, X, sc, X_train, X_test, Y_train, Y_test, model4, y_pred
    df = df.drop(['target', ], axis=1)
    # st.write(df.shape)
    # Target variable and train set
    Y = df[['Attack Type']]
    X = df.drop(['Attack Type', ], axis=1)
    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    # Split test and train data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    # st.write(X_train.shape, X_test.shape)
    # st.write(Y_train.shape, Y_test.shape)
    model4 = SVC(gamma='scale')
    start_time = time.time()
    model4.fit(X_train, Y_train.values.ravel())
    end_time = time.time()
    Training_time = end_time-start_time
    st.write("Training time: ", end_time-start_time)
    start_time = time.time()
    Y_test_pred4 = model4.predict(X_test)
    end_time = time.time()
    Testing_time = end_time-start_time
    st.write("Testing time: ", end_time-start_time)
    # Create a dictionary of values for the x and y axes
    data = {"Training": Training_time, "Testing": Testing_time}
    # Create a bar chart using the st.bar_chart() function
    st.bar_chart(data)
    # Add labels and titles to the bar chart using Matplotlib
    fig, ax = plt.subplots(figsize=(10,10), dpi=100)
    ax.bar(data.keys(), data.values())
    ax.set_xlabel("Process")
    ax.set_ylabel("Time (s)")
    ax.set_title("Training vs Testing Times")
    st.write("Train score is:", model4.score(X_train, Y_train))
    st.write("Test score is:", model4.score(X_test, Y_test))
    # Create a dictionary of values for the x and y axes
    data = {"Train Score": model4.score(X_train, Y_train), "Test Score": model4.score(X_test, Y_test)}
    # Create a bar chart using the st.bar_chart() function
    st.bar_chart(data)
    # Add labels and titles to the bar chart using Matplotlib
    fig, ax = plt.subplots(figsize=(10,10), dpi=100)
    ax.bar(data.keys(), data.values())
    ax.set_xlabel("Process")
    ax.set_ylabel("Time (s)")
    ax.set_title("Train vs Test Scores")
    # Evaluate the model on the testing set
    y_pred = model4.predict(X_test)
    # st.write(y_pred)
    return y_pred

# ----------------------------------------------------------------------------------------------------------------
# If the user has uploaded a file, process it
if uploaded_file is not None:
    # Load the data from the CSV file
    df = pd.read_csv(uploaded_file)
    # Use st.markdown() with variable to display text with custom font size
    st.markdown("<h1 style='text-align: left; font-size: 30px; '> File Details : </h1>",unsafe_allow_html=True)
    st.markdown("<span style='margin-left:30px;'><span>File Name :<span> {}</span>".format(uploaded_file.name), unsafe_allow_html=True)
    st.markdown("<span style='margin-left:30px;'><span>File Type :<span> {}</span>".format(uploaded_file.type), unsafe_allow_html=True)
    st.markdown("<span style='margin-left:30px;'><span>File Size :<span> {}</span>".format(uploaded_file.size), unsafe_allow_html=True)
    st.write(df.head(10))
    # Train and test the SVM model
    report = train_test_model(df)
    # Display the classification report
    st.header('Classification Report')
    display_classification = classification_report(Y_test, y_pred)
    st.text(display_classification)
    value = {0 : 'DoS Attack', 1 : 'Normal', 2 : 'Probe Attack', 3 : 'R21 Attack', 4 : 'U2R Attack'}
    key_list = list(value.keys())
    percen_list = []
    for i in key_list:
        count = np.count_nonzero(report == key_list[i]) 
        percentage = (count / len(report)) * 100 
        percen_list.append(percentage)
        if(value[i] == 'Normal'):
            st.success(f"{value[i]}_____________________________________________________________________________{percentage:.2f}%\n\n\n\n\n")  
        else:
            st.error(f"{value[i]}_____________________________________________________________________________{percentage:.2f}%\n\n\n\n\n")  
    my_dict = {'Dos' : percen_list[0], 
               'Normal' : percen_list[1],
               'Probe' : percen_list[2],
               'R21' : percen_list[3],
               'U2R' : percen_list[4]}

    # Compute the total count of values in the dictionary
    total = sum(my_dict.values())
    # Compute the percentage values for each key
    percentages = {k: v/total for k, v in my_dict.items()}
    # Plot the bar chart of percentages
    st.bar_chart(percentages)
# --------------------------------------------------------------------------------------------------------------------------
    # Add a title to the page
    st.title('Uplaod CSV File For PREDICTION')
    # Add a file uploader to allow the user to select a CSV file
    Predict_file = st.file_uploader('Upload a CSV file', key="file_uploader1")
    if Predict_file is not None:
    # Load the data from the CSV file
        pred = pd.read_csv(Predict_file)
        result = model4.predict(pred)
        # st.write(result)
        st.write(pred.head(10))
        # Display the classification report
        st.header('Classification Report')
        i = len(result)
        for x in range(0,i):
            st.write("Result {} : ".format(x))
            if (result[x] == 1):
                st.success("NO INTRUSION DETECTED...")
            else:
                st.error("INTRUSION DETECTED...")
        value = {0 : 'DoS Attack', 1 : 'Normal', 2 : 'Probe Attack', 3 : 'R21 Attack', 4 : 'U2R Attack'}
        key_list = list(value.keys())
        percen_list = []
        for i in key_list:
            count = np.count_nonzero(result == key_list[i]) 
            percentage = (count / len(result)) * 100 
            percen_list.append(percentage)
            st.write(f"{value[i]}_____________________________________________________________________________{percentage:.2f}%\n\n\n\n\n")    
        my_dict = {'Dos' : percen_list[0], 
                'Normal' : percen_list[1],
                'Probe' : percen_list[2],
                'R21' : percen_list[3],
                'U2R' : percen_list[4]}

        # Compute the total count of values in the dictionary
        total = sum(my_dict.values())
        # Compute the percentage values for each key
        percentages = {k: v/total for k, v in my_dict.items()}
        # Plot the bar chart of percentages
        st.bar_chart(percentages)
    else:
        st.write('Please upload a CSV file to Start The Prediction.')
else:
    st.write('Please upload a CSV file to get started.')
