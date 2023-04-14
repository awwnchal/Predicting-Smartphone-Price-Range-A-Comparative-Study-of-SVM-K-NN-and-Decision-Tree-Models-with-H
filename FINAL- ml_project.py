#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 13:52:15 2023

@author: Murad, Winnie, Anchal
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb

df = pd.read_csv('/Users/anchalchaudhary/Downloads/mobile_price.csv')

# =============================================================================
# #EDA
# =============================================================================
df.shape
## display the summary statistics of the dataset
df.describe()
## display the first five rows of the dataset
df.head()
## display info
df.info()
## check if there is missing value
df.isnull().sum()
# plot a heatmap to show the correlations
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(26, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show() ## As we can see, ram has the highest correlation with price_range

## scatterplot: ram vs price_range
# Plot a scatter plot between 'price_range' and 'ram'
plt.scatter(df['ram'], df['price_range'])

# Label the x-axis and y-axis with appropriate names
plt.xlabel('RAM')
plt.ylabel('Price Range')

# Show the plot
plt.show()

## create scatterplot for all variables vs price_range
variables = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi']

for i in variables:
    plt.scatter(df[i], df['price_range'])

    # Label the x-axis and y-axis with appropriate names
    plt.xlabel(i)
    plt.ylabel('Price Range')   

    # Show the plot
    plt.show()
    
  ## price_range
s = df["price_range"].value_counts()
plt.pie(s, labels = s.index, autopct='%1.1f%%')
plt.show()  ## We can see each of the price_range is 25%, the given data is balanced.

### binary variables

binary_variables = ['blue','dual_sim','four_g', 'three_g','touch_screen','wifi']

for i in binary_variables:
## price_range
    count_1 = df[i].value_counts()
    plt.pie(count_1, labels = count_1.index, autopct='%1.1f%%')
    plt.title(f'Distribution of {i}')
    plt.show()
    
# =============================================================================
# #Linear Regression 
# =============================================================================
X=df.drop('price_range',axis=1)
y=df['price_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
linear_model = LinearRegression().fit(X_train,y_train)
linear_pred = linear_model.predict(X_test)
pd.DataFrame({"Actual y": y_test, "Predicted y": linear_pred})


linear_r2 = r2_score(y_test, linear_pred)
print(f"R_squared for Linear Regression: {linear_r2:.3f}") 
linear_mse = mean_squared_error(y_test, linear_pred)
print(f"MSE for Linear Regression: {linear_mse:.3f}")
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))
print(f"RMSE for Linear Regression: {linear_rmse:.3f}")

print('***********************************************************************************************************')



# =============================================================================
# #SVM
# =============================================================================

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the dataset into training and testing sets: 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 1)
# Train an SVM model on the training set
svm_model = SVC(kernel='linear', random_state=1)
svm_model.fit(x_train, y_train)

# Predict the labels of the test set using the trained SVM model
y_pred = svm_model.predict(x_test)
# Compute the accuracy score of the SVM model on the test set
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy) #0.96

# =============================================================================
# K-Nearest neighbour -  used the elbow method to find the optimal value of k
# =============================================================================
X = df.drop('price_range', axis=1)
y = df['price_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

#Using Elbow method

# Calculate the accuracy score for different values of k
k_values = list(range(1, 21))
accuracy_scores = []

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, knn_pred)
    accuracy_scores.append(accuracy)

# Plot the results
plt.plot(k_values, accuracy_scores)
plt.xlabel('Number of neighbors (k)')
plt.ylabel('Accuracy score')
plt.title('KNN model performance for different values of k')
plt.show()

#got the value of k as 9 from the elbow method
# Train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=9)
knn_model.fit(X_train, y_train)

# Make predictions on the testing data
knn_pred = knn_model.predict(X_test)

#now, evaluating the performance of the model

#The accuracy score measures the proportion of correct
#predictions made by the model, while the confusion matrix
#shows the number of true positives, true negatives, false positives, and 
#false negatives for each class. The classification report shows various metrics 
#such as precision, recall, and F1-score for each class, as well as the macro 
#and weighted averages of these metrics.
accuracy = accuracy_score(y_test, knn_pred)
print(f"Accuracy score for KNN model: {accuracy:.3f}")

confusionmatrix = confusion_matrix(y_test, knn_pred)
print(f"Confusion matrix for KNN model:\n{confusion_matrix}")

classifreport = classification_report(y_test, knn_pred)
print(f"Classification report for KNN model:\n{classifreport}")

print('***********************************************************************************************************')

# =============================================================================
# #Decision Tree
# =============================================================================

# Training a decision tree classifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

#Making predictions on the testing data
y_pred = dtc.predict(X_test)

#now, evaluating the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

report = classification_report(y_test, y_pred)

print("Classification Report:\n", report)
#Precision and recall metrices should be high as possible. They gives us accuracy level out of different classes.
#We use f1-score to compare models that have different precision and recall levels.



#performing  hyperparameter tuning and feature engineering on the "Mobile Price Classification" 
#dataset using scikit-learn library
# Feature engineering: create a new feature for screen size
df['screen_size'] = df['px_width'] * df['px_height']

# Split the dataset into training and testing sets
X = df.drop('price_range', axis=1)
y = df['price_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

# Define the hyperparameters to tune
params = {
    'max_depth': [3, 5, 7, 9, 11],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': [None, 'sqrt', 'log2']
}

#Creating a decision tree classifier
dtc = DecisionTreeClassifier()

#finding best hyperparameters
#10 fold cross validation
grid_search = GridSearchCV(dtc, params, cv=10, n_jobs=-1)
grid_search.fit(X_train, y_train)

# now,training the decision tree classifier with the best hyperparameters
best_dtc = grid_search.best_estimator_
best_dtc.fit(X_train, y_train)

# predictions on the test data
y_pred = best_dtc.predict(X_test)

# evaluating the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

#We noticed that accuracy, precision and recall improved after hyperparameter tuning and feature engineering

print('***********************************************************************************************************')

# =============================================================================
# Principal Component Analysis
# =============================================================================

df = pd.read_csv('/Users/anchalchaudhary/Downloads/mobile_price.csv')

#defining x and y 
X = df.drop('price_range', axis=1)
y = df['price_range']

#Standardizing the features(important before PCA)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)


# Performing PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)


# Plotting the explained variance ratio
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.show()
#Based on the plot, we can see that the first two principal components 
#explain a significant portion of the variance in the data, which is 
#why we chose n_components=2.


# Creating a new DataFrame with the PCA components and target variable
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['price_range'] = y

# Plotting the data points in the first two principal components

sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='price_range')
plt.show()
#Each point in the scatterplot represents a mobile phone from 
#the dataset, and the color of the point corresponds to its price range. By 
#visualizing the data in this way, we can see how well the PCA has separated
#the mobile phones into different price ranges.

print('***********************************************************************************************************')
# =============================================================================

# #Lasso Regression for important feature selection
# 10 fold cross validation to decide lamba 
# =============================================================================

df = pd.read_csv('/Users/anchalchaudhary/Downloads/mobile_price.csv')
X = df.drop('price_range', axis=1)
y = df['price_range']

#Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

#creating the Lasso Regression model with 10 fold cross-validation
lasso_model = LassoCV(cv=10)
# Fitting the model on the training data
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test) #getting predictiond

pd.DataFrame({"Actual y": y_test, "Predicted y": lasso_pred})

#Printing the R-squared and mean squared error
lasso_r_sqaure = lasso_model.score(X_test, y_test)
print(f"R_squared for Lasso Regression: {lasso_r_sqaure:.3f}")
lasso_mse = mean_squared_error(y_test, lasso_pred)
print(f"MSE for Lasso Regression: {lasso_mse:.3f}")
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
print(f"RMSE for Lasso Regression: {lasso_rmse:.3f}")
# Print the optimal value of lambda
print(f"Optimal value of lambda: {lasso_model.alpha_:.3f}")



# the coefficients of the features
lasso_coefs = pd.DataFrame({"Feature": X_train.columns, "Coefficients": lasso_model.coef_})
print(lasso_coefs)

# displaying the important variables selected by Lasso(feature selection)
important_vars = lasso_coefs.loc[lasso_coefs["Coefficients"] != 0, "Feature"]
print("Important variables selected by Lasso:")
print(important_vars)
print('***********************************************************************************************************')

# =============================================================================

# #Random Forest - we use the selected subset of features from the Lasso 
# regression to train and test the random forest, and also uses cross-validation 
# to tune the hyperparameters of the random forest model. Additionally,
#  we've added Adaboost as another classifier to compare against.
# =============================================================================

# Create a new DataFrame with only the important variables
X_lasso = X_train.loc[:, important_vars]
# Perform train-test split on the new DataFrame
X_train_lasso, X_test_lasso, y_train_lasso, y_test_lasso = train_test_split(X_lasso, y_train, test_size=0.33, random_state=101)

    
# Random forest with cross-validation
rfc = RandomForestClassifier(random_state=101)
param_dist = {'n_estimators': [100, 200, 300, 400, 500],
              'max_depth' : [10, 20, 30, 40, 50],
              'criterion' :['gini', 'entropy']}
rfc_cv = RandomizedSearchCV(rfc, param_distributions=param_dist, cv=10, n_iter=10, random_state=101)
rfc_cv.fit(X_train_lasso, y_train_lasso)

# Fitting & predict using the best model from cross-validation
rfc_best = rfc_cv.best_estimator_
rfc_best.fit(X_train_lasso, y_train_lasso)
rfc_pred = rfc_best.predict(X_test_lasso)

# Evaluate model performance
print(f"Accuracy Score for Random Forest: {accuracy_score(y_test_lasso, rfc_pred):.3f}")
print(f"Confusion Matrix for Random Forest: \n{confusion_matrix(y_test_lasso, rfc_pred)}")
print(f"Classification Report for Random Forest: \n{classification_report(y_test_lasso, rfc_pred)}")

# Adaboost
abc = AdaBoostClassifier(base_estimator=rfc_best, n_estimators=50, learning_rate=1, random_state=101)
abc.fit(X_train_lasso, y_train_lasso)
abc_pred = abc.predict(X_test_lasso)

# Evaluate model performance
print(f"Accuracy Score for Adaboost: {accuracy_score(y_test_lasso, abc_pred):.3f}")
print(f"Confusion Matrix for Adaboost: \n{confusion_matrix(y_test_lasso, abc_pred)}")
print(f"Classification Report for Adaboost: \n{classification_report(y_test_lasso, abc_pred)}")
print('***********************************************************************************************************')

#Adaboost was used to boost the performance of
# the random forest classifier by treating it 
# as the base estimator. This is a common approach, 
# as random forests are already strong learning algorithms
# that can be further improved with boosting. 
#However, we didn't see any improvement in the model performance after adding Adaboost. The accuracy remains the 
#same.

# =============================================================================
# # XGBoost
# =============================================================================
dtrain = xgb.DMatrix(X_train_lasso, label=y_train_lasso)
dtest = xgb.DMatrix(X_test_lasso, label=y_test_lasso)

#setting  parameters
params = {'objective': 'multi:softmax', 
          'num_class': 4, 
          'max_depth': 5, 
          'eta': 0.1, 
          'subsample': 0.8, 
          'colsample_bytree': 0.8}

#Cross-validation to find optimal number of boosting rounds
cv_results = xgb.cv(dtrain=dtrain, params=params, nfold=10, num_boost_round=1000, early_stopping_rounds=10, metrics='merror', seed=1000)
num_boost_rounds = cv_results.shape[0]

# Training XGBoost model
model = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_rounds)

#prediction and performance
xgb_pred = model.predict(dtest)
xgb_pred = [int(round(pred)) for pred in xgb_pred]
print(f"Accuracy Score for XGBoost: {accuracy_score(y_test_lasso, xgb_pred):.3f}")
print(f"Confusion Matrix for XGBoost: \n{confusion_matrix(y_test_lasso, xgb_pred)}")
print(f"Classification Report for XGBoost: \n{classification_report(y_test_lasso, xgb_pred)}")

#not so impressive accuracy from Xg boost
print('***********************************************************************************************************')

#Let's see the performace of all the models we have created so far 
models = ['Linear reg', 'SVM','KNN', 'Decisiontree ', 'lasso ','randomF', 'Xgboost']
acc_scores = [0.913, 0.96, 0.92, 0.859,0.86,0.90,0.892]

plt.bar(models, acc_scores, color=['lightblue', 'pink', 'lightgrey', 'cyan'])
plt.ylabel("Accuracy scores")
plt.title("Which model is the most accurate?")
plt.show()
#Clearly, SVM outperforms all other models with an accuracy of 0.96. There can
#be multiple reasons for it outperforming other models
#such as SVM can handle both linear and non linear relationships and 
#it is less prone to overfititng.





