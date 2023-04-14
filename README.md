

# Predicting Smartphone Price Range:
# A Comparative Study of SVM, K-NN, and Decision Tree Models with Hyperparameter Tuning and Feature Engineering


# Executive Summary

With a wide range of mobile phones available in the market, choosing the one that is in
accordance with the budget can be a complicated task for customers. It is also a challenge for
businesses to set the price of mobile phones and remain competitive in the market. Therefore,
our project aims to develop a predictive model that can help determine the approximate budget
for a mobile phone based on different relevant factors and help the business to set their price of
mobile phones. In this project, we compared the performance of many machine learning
algorithms - Linear Regression, Support Vector Machine (SVM), K-Nearest Neighbors (K-NN),
and Decision Tree - with hyperparameter tuning and feature engineering. We also explored the
performance of Random Forest(with Adaboost), Lasso Regression, and XGBoost. Our report
provides insights into the performance of different machine learning algorithms for mobile phone
price range prediction. These insights/findings can aid future researchers and developers in
selecting appropriate algorithms for similar tasks.

# Background, Context, and Domain Knowledge

As technology is changing rapidly nowadays, mobile phones have become an essential
part of our daily life. We use mobile phones in many aspects, such as accessing social media
platforms, playing games, and even making payments. Mobile phones became a crucial
connection between each individual and the outside world, serving as a gateway to vast
amounts of information. The number of mobile phone manufacturers is increasing rapidly and
the market for mobile phones has thousands of options, from budget-friendly to premium price
mobile phones. In fact, there are a lot of factors that impact the price of mobile phones, for
example, random access memory (RAM), which is used to store data in mobile phones. In this
analysis report, we aim to develop a machine learning model that can be used to accurately
predict mobile phone prices based on a variety of factors. A precise and robust predictive model
can help customers to determine the approximate budget for a mobile phone based on different
factors, and also help businesses to set the price of mobile phones and remain competitive in
the market.
The dataset contains thousands of observations of features about mobile phones and
their related price range. It includes 20 factors that can impact the price of mobile phones:
battery power, blue, clock speed, dual sim, front camera pixels, 4G, 3G, internal memory, mobile
depth, weight, cores of processor, primary camera pixels, pixel resolution height, pixel resolution
width, RAM, screen height, screen width, battery time, touch screen and wifi. Our target variable
would be the price range, which contains values of 0 (low cost), 1 (medium cost), 2 (high cost),
and 3 (premium cost) to indicate the prices of mobile phones.
Discussion/ Alignment with Business Model
Traditionally, firms have attempted to predict and set the mobile price through different
methods, for example, conducting market research. Firms used surveys to gather information on
consumer preferences and purchase behavior. By understanding customers’ need and
willingness to pay, the firms can make more accurate predictions on mobile prices and optimize
their pricing strategy. However, this process of market research is time-consuming, costly and
inefficient. To solve this problem, we can develop a predictive model that uses a combination of
historical data and machine learning algorithms to predict the mobile price with different factors.
It aligns with the business model of the mobile industry and provides a reliable and effective tool
for making pricing decisions, which help the firms have better decision making about the pricing
strategies.
From the customer perspective, customers conduct research and compare prices across
different brands and models traditionally, which is their strategy to look for discounts and
promotions to get the best deal possible. This approach has limitations such as time-consuming,
limited resources, and inaccurate information. To solve this problem, an accurate and robust
predictive model can help customers to determine the estimated price for a mobile phone based
on their needs of specifications and allow them to save the budget accordingly.

# Project Analysis & Insights

We began our analysis by building a linear regression model, which had an accuracy
score of 0.91 and a root mean square error of 0.32. While this is a decent score, it is important
to note that linear regression models are best suited for data that has a linear relationship
between the independent and dependent variables. Linear regression is also sensitive to
outliers, which can negatively impact the model's performance. As a result, linear regression
may not do well in the dataset according to the linearity and outliers issue.
Next, we used a support vector machine (SVM) model, which yielded an accuracy score
of 0.96, which was higher than the accuracy score we got using linear regression. SVM is able
to handle both linear and non-linear relationships between the independent and dependent
variables, and is robust to outliers and overfitting. These are the reasons why SVM performed
well with the dataset, yielding an accuracy score of 0.96.
We also performed a K-Nearest Neighbors (KNN) model to predict the mobile phone
price range. We used the elbow method to determine the optimal value of k, which was 9 in our
case. The accuracy score we got from the KNN model is 0.92, which is higher than linear
regression. While the performance of the KNN model was not bad, it was not as impressive as
SVM. KNN models are suitable for data with non-linear relationships, but they can be
computationally expensive and sensitive to the value of k.
After that, decision trees are simple to understand and interpret, but they tend to overfit
the data, which can negatively impact their performance. We then evaluated the performance of
the decision tree on the test data and found that the accuracy was low compared to the previous
models (0.859). We used accuracy score and classification report as metrics to evaluate the
model's performance. To dig deeper and improve the model's performance, we proceeded with
hyperparameter tuning and feature engineering on the "Mobile Price Classification" dataset
using scikit-learn library. Feature engineering involved creating a new feature for screen size
(df['screen_size'] = df['px_width'] * df['px_height']). Hyperparameter tuning involved adjusting the
model's parameters to optimize its performance on the test data. Often, raw data contains
features that are not directly useful for modeling or have high correlation with other features,
which can lead to overfitting. In such cases, feature engineering can help in creating new
features that can capture the underlying patterns or relationships in the data that are not
explicitly represented by the raw features. After fitting the model, we noticed that accuracy,
precision, and recall improved after hyperparameter tuning and feature engineering. Therefore,
feature engineering is a crucial step in machine learning and can significantly impact the
accuracy and performance of predictive models.
To further investigate which features were the most important out of all the features in
the original dataset, we decided to use Lasso regression, which is known for its feature
selection capabilities. The R-squared value of the Lasso regression was 0.91, indicating that it is
a good fit for the data. The selected features from the Lasso regression were battery power,
mobile_wt, px_height, px_width, and Ram, which are the five most important features that
impact the price of mobile phones. We then trained and tested a random forest model using the
selected features from the Lasso regression. The accuracy score we got was 0.90. We also
used Adaboost to improve the performance of the random forest classifier, but it did not provide
any significant improvement over the base model. Adaboost was used to boost the performance
of the random forest classifier by treating it as the base estimator. This is a common approach,
as random forests are already strong learning algorithms that can be further improved with
boosting. However, we didn't see any improvement in the model performance after adding
Adaboost. The accuracy remains the same.
Lastly, we tried XGBoost, which is a powerful boosting algorithm that is suitable for
high-dimensional datasets. While XGBoost had a decent accuracy score of 0.89, it did not
outperform SVM.

In summary, each model has its own strengths and weaknesses. Linear regression is
best suited for data with linear relationships, while SVM can handle both linear and non-linear
relationships and is robust to outliers and overfitting. KNN models are suitable for non-linear
data, but can be computationally expensive. Decision trees are simple to understand and
interpret, but tend to overfit the data. Random forests are robust to overfitting and are suitable
for high-dimensional data, but Adaboost did not significantly improve the performance of the
model. XGBoost is a powerful boosting algorithm that is suitable for high-dimensional data, but it
did not outperform SVM in our analysis. For the accuracy score overall, SVM has the highest
accuracy among these models.

# Recommendations and Business Value

After analyzing the dataset and comparing different models, we have developed a
machine learning model that can accurately predict mobile phone prices based on a variety of
factors. This can provide value to both customers and manufacturers in the following ways:
1. For customers, our predictive model can help determine the approximate budget for a
mobile phone based on different factors(such as battery_power, mobile_wt, px_height,
px_width, ram). This can save time and effort in the decision-making process and also
help customers make informed decisions when purchasing a mobile phone. For
instance, suppose a customer is looking to buy a mobile phone with high RAM and a
large display but has a limited budget. In that case, the customer can use our predictive
model by inputting his desired specifications and obtain an approximate price range.
This can help the customer make a more informed decision about which mobile phone to
purchase, making his life easier.
2. For manufacturers, our predictive model can help in setting the price of mobile phones
and remain competitive in the market. By analyzing the factors that impact the price of
mobile phones, businesses can make informed decisions about their pricing strategy and
adjust prices accordingly. We have done feature selection in order to find out the major
factors(i.e. important variables) that explain the most variance in the prices of mobile
phones. Our predictive model can also be used by retailers to optimize their pricing
strategy. By analyzing the data, retailers can identify which factors contribute the most to
the price range and adjust their pricing strategy accordingly. This can lead to increased
sales and revenue for the retailer.
In addition to that, we can also make the following recommendations for businesses:
1. To Focus more on RAM: We found out through our analysis that RAM has the highest
correlation with the price range of mobile phones. Therefore, businesses should focus on
providing mobile phones with higher RAM to appeal to a large customer base looking for
higher-priced mobile phones.
2. Improving battery life: Battery power and battery time were also found to be important
factors that impact the price of mobile phones. Therefore, businesses will benefit from
focusing on providing mobile phones with better battery life to attract customers who
prioritize battery life.
3. Keeping up with trends: As technology is changing rapidly, businesses should keep up
with the latest trends and provide mobile phones with the latest features and
technologies. This can help them remain competitive in the current market and attract
customers who are looking for the latest features in their mobile phones.

# Summary and Conclusions

In this project, we have performed exploratory data analysis and developed machine
learning models to predict the price range of mobile phones. We have used several algorithms,
including SVM, KNN, and decision tree classifiers, random forest(Adaboost), lasso regression,
Xgboost and evaluated their performance using accuracy score, confusion matrix, and
classification report. Our results showed that SVM performed better than other models,
achieving an accuracy score of 0.96. We also used hyperparameter tuning and feature
engineering to improve the performance of our models.
In conclusion, we have successfully developed a machine learning model that can
accurately predict the price range of mobile phones. However, there is always room for
improvement, and we recommend further exploration of other algorithms and techniques to
enhance the performance of our models. In Addition to that, we prefer collecting more data to
increase the diversity and size of the dataset, which could lead to better predictions and more
accurate insights.


References

Data sources:
Kaggle. (n.d.).
https://www.kaggle.com/code/vikramb/mobile-price-prediction/input?select=train.csv
Retrieved from https://www.kaggle.com
