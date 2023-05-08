# Intrusion-Detection-System
- IDS monitors a network or systems for malicious activity and protects a computer network from unauthorized access from users,including perhaps insider.
- The motive of this study is to propose a predictive model (i.e. a classifier) capable of distinguishing between 'bad connections' (intrusions/attacks) and a 'good (normal) connections' after applying some feature extraction on KDD Cup 1999 dataset by DARPA. 

### Intrusion-Detection-System [Code](https://github.com/anupam215769/Movie-Recommender-System-ML/blob/main/movie-recommender-system.ipynb) OR <a href="https://colab.research.google.com/github/hacker-404-error/INTRUSION_DETECTION_-SYSTEM/blob/master/.ipynb_checkpoints/main-checkpoint.ipynb#scrollTo=MTHedt0aiG2Tb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>


## How To Run (Graphical Interface/In Web Browser)

> Note - Install streamlit library before running the code

```
pip install streamlit
```

1. Download all the files and put them in a same folder

2. Open APP.py using any python compiler

3. Run the app.py

4. Then type `streamlit run app.py` in the terminal

5. This project will open in your web browser (as shown in the screenshot above)

## How To use The Streamlit app for IDS
1. Upload the Data file For Test and Train the Data
   
![rec](https://raw.githubusercontent.com/hacker-404-error/INTRUSION_DETECTION_-SYSTEM/master/Images/1.%20Select%20CSV%20file%20For%20data%20in%20App.png)
   
2. After Uploading It will show the details of the data and the data itself in rows and columns and test and train the data automatically
   
![rec](https://raw.githubusercontent.com/hacker-404-error/INTRUSION_DETECTION_-SYSTEM/master/Images/2.%20Training%20And%20Testing%20.png)

3. It will automatically generate the training and testing time as well as score

![rec](https://raw.githubusercontent.com/hacker-404-error/INTRUSION_DETECTION_-SYSTEM/master/Images/3.%20Timings.png)

4. Classification Report For the above data

![rec](https://raw.githubusercontent.com/hacker-404-error/INTRUSION_DETECTION_-SYSTEM/master/Images/4.%20Classification%20Report.png)

5. Now Upload the data You want to Predict

![rec](https://raw.githubusercontent.com/hacker-404-error/INTRUSION_DETECTION_-SYSTEM/master/Images/5.%20Upload%20File%20For%20Prediction.png)

6. Classification Report For the Predicted data given
   
![rec](https://raw.githubusercontent.com/hacker-404-error/INTRUSION_DETECTION_-SYSTEM/master/Images/6.%20Classification%20Report%20for%20Prediction.png)
   
![rec](https://raw.githubusercontent.com/hacker-404-error/INTRUSION_DETECTION_-SYSTEM/master/Images/7.%20Classification%20Report%20for%20Prediction2.png)

## How To Run (In Jupyter Notebook)

> Note - Install Jupyter Notebook

```
pip install jupyter-lab
```

1. Open Jupyter Notebook using CMD by typing `jupyter-lab`

2. Now, locate the folder of this project

3. Open `main.ipynb` and run all the cells

4. At the last Cell : Its a Tkinter Application which take user input about the network data, Enter it.

5. You will get the prediction that given network data from a website is malacious or not




# DATASET
KDD Cup 1999 dataset by DARPA
The whole dataset can be downloaded from- http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

# MODELS
- A total of seven models is trained and tested. 
- The performance of all the algorithms is examined based on accuracy and computational time. 
- Derived results show that Support Vector Machine outperforms the best on measures like Accuracy and Computational Time.

# ALGORITHMS USED
Gaussian Naive Bayes, Decision Tree, Random Forest, SVM, Logistic Regression,Gradient Boosting, ANN



