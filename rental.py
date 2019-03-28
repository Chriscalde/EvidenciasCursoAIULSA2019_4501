#imports

import numpy
import pandas

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

import matplotlib.pyplot as plt

#reading the csv

rental_df = pandas.read_json('train.json')

#analisis 
print(rental_df.head(10))

print(rental_df.dtypes)

print(rental_df.apply(lambda x: x.isnull().any()))

print(rental_df.describe())
#boxplot usage


bathroom_data = rental_df['bathrooms']
bedroom_data = rental_df['bedrooms']
price_data = rental_df['price']
# plt.boxplot(price_data)
# plt.show()
#data preparation

# outliers 

bathroom_upper_limit = 8
bathroom_lower_limit = 1
bedroom_upper_limit = 5.5
price_upper_limit = 50000

bathroom_data.loc[rental_df['bathrooms']>bathroom_upper_limit] = bathroom_upper_limit
bedroom_data.loc[rental_df['bedrooms']>bedroom_upper_limit] = bedroom_upper_limit
price_data.loc[rental_df['price']>price_upper_limit] = price_upper_limit

bathroom_data.loc[rental_df['bathrooms']<bathroom_lower_limit] = bathroom_lower_limit

# plt.boxplot(price_data)
# plt.show()

# # processing the data

processed_df = rental_df.drop(['features','photos','description','display_address','street_address'],axis=1)

processed_df['building_id'] = rental_df.building_id.astype('category')
processed_df['created'] = rental_df.created.astype('category')
processed_df['listing_id'] = rental_df.listing_id.astype('category')
processed_df['manager_id'] = rental_df.manager_id.astype('category')
processed_df['building_id'] = pandas.get_dummies(processed_df['building_id'])
processed_df['created'] = pandas.get_dummies(processed_df['created'])
processed_df['manager_id'] = pandas.get_dummies(processed_df['listing_id'])
print(processed_df.head(5))

# # separate classes 
data_features = processed_df.drop(['interest_level'], axis=1).values
data_labels = processed_df['interest_level'].values

# # univariate selection 
# print(data_features[:10])

test = SelectKBest(score_func=f_classif,k=4)
fit = test.fit(data_features,data_labels)
numpy.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(data_features)
print(features[0:10])

# # recursive features 

# model = LogisticRegression()
# rfe = RFE(model,4)
# fit = rfe.fit(data_features,data_labels)
# print("Num Features:")
# print(fit.n_features_)
# print("Selected Features:")
# print(fit.support_)
# print("Feature Ranking:")
# print(fit.ranking_)

# # principal component analysis

# pca = PCA(n_components=4)
# fit = pca.fit(data_features)
# print("Explained Variance:")
# print(fit.explained_variance_ratio_)
# print(fit.components_)

# # feature importance 

# model = ExtraTreesClassifier()
# model.fit(data_features,data_labels)
# print(model.feature_importances_)

# # MinMaxScaler

# scaler = MinMaxScaler()
# scaler.fit(data_features)
# features = scaler.fit_transform(data_features)

# # StandardScaler

scaler = StandardScaler()
features = scaler.fit_transform(data_features)

# KMeans 
# X= numpy.array(data_features)
kmeans = KMeans(n_clusters=2,random_state=0).fit(features)
# features = KMeans.transform(X)
# #implement decision tree

X_train, X_test, Y_train, Y_test = train_test_split(features, data_labels, test_size=0.3)

# clf = DecisionTreeClassifier()

# clf.fit(X_train,Y_train)

# score = clf.score(X_test,Y_test)

# print(score)

# implement classifier comparison

names = ["Decision Tree Classifier", "MLP Classifier", "Random Forest Classifier", "AdaBoost", "Bagging Classifier"]

classifiers = [
    DecisionTreeClassifier(max_depth=5),
    MLPClassifier(alpha=1),
    RandomForestClassifier(max_depth=5, max_features=1, n_estimators=10),
    AdaBoostClassifier(n_estimators=10),
    BaggingClassifier(max_features=1,n_estimators=10)
]

for name, clf in zip(names, classifiers):
    clf.fit(X_train,Y_train)
    score = clf.score(X_test,Y_test)
    y_pred = clf.predict(X_test)
    print(name+": "+str(score))
    print(confusion_matrix(Y_test,y_pred,labels=None))
    print(cohen_kappa_score(Y_test,y_pred, labels=None))
    print(classification_report(Y_test,y_pred,labels=None))
#     print(cross_val_score(clf,X_test,y_pred,cv=8))

# evaluation methods
# confusion matrix

# y_true = rental_df['interest_level'].values
# y_pred = data_labels
# print(confusion_matrix(y_true,y_pred,labels=None))
