"""
Pima Indians Diabetes Data Set
Free from: https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes

Information on the dataset is found here:
https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.names

This is where we get the list of features and learn that the last column "class" is
a 1 if the persion tested positive for diabetes, or 0 if no diabetes.

"""

import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Load Data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
names = ['pregnancies', 'plasma', 'blood_pressure', 'tri_skin_fold', 'insulin', 'bmi', 'pedigree', 'age', 'class']
data = pd.read_csv(url, names=names)

#Visualize/Play with Data
#print "Data Loaded, now hit enter to view summary of data"
#raw_input("Press Enter to Continue..")
#print data.head()
#raw_input("Press Enter to Continue..")
#print data.shape
#raw_input("Press Enter to Continue..")
#print data.describe()
#raw_input("Press Enter to Continue..")
#scatter_matrix(data)
#plt.show()

#"Pre-Process Data"
data_array = data.values
features = data_array[:,0:8]
labels = data_array[:,8]
#StandardScaler() will fit the data to look more like a normal distribution
scaler = StandardScaler().fit(features)
features_scaled = scaler.transform(features)

#Split the Data into Train & Test set
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.25)
#This will split the data so that 25% is in our test set
#X are the features, y is the label (what we are trying to predict)


#KFold will divide the training set into n groups of samples
#It will then use n-1 groups as training set and the left out group as the validation set
#This can then be used to save data but allow for tuning of your model and testing how reliable the model is
kfold = KFold(n_splits=10, random_state=0)
model = LogisticRegression()
results = cross_val_score(model, X_train, y_train, cv=kfold)
#View the results of each pass
print "K Folds Results:"
print results
#View the average of the results
print "K Folds Mean: "
print results.mean()

#Make a simple LogisticRegression 
model.fit(X_train, y_train)
#See how accurate our model is
print "Logistic Regression Score on Test Set:"
print model.score(X_test, y_test)