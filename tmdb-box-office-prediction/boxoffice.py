#imports

import numpy as np
import pandas as pd

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report
from sklearn.model_selection import cross_val_score

import ast

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

import matplotlib.pyplot as plt

# reading the csv 


boxoffice_df = pd.read_csv('train.csv', index_col=None)

dict_columns = ['genres']

def text_to_dict(df):
    for column in dict_columns:
        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )
    return df

boxoffice_df = text_to_dict(boxoffice_df)
#pre-analisis

print(boxoffice_df.head(10))
print(boxoffice_df.dtypes)

#genre filling
for i, e in enumerate(boxoffice_df['genres'][:5]):
    print(i, e)
    
list_of_genres = list(boxoffice_df['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
print(Counter([i for j in list_of_genres for i in j]).most_common())

boxoffice_df['num_genres'] = boxoffice_df['genres'].apply(lambda x: len(x) if x != {} else 0)
boxoffice_df['all_genres'] = boxoffice_df['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
top_genres = [m[0] for m in Counter([i for j in list_of_genres for i in j]).most_common(15)]
for g in top_genres:
    boxoffice_df['genre_' + g] = boxoffice_df['all_genres'].apply(lambda x: 1 if g in x else 0)
    
boxoffice_df = boxoffice_df.drop(['genres'], axis=1)

#outliers

budget_data = boxoffice_df['budget']
popularity_data = boxoffice_df['popularity']
# plt.boxplot(popularity_data)
# plt.show()

budget_upper_limit = 200000000
popularity_upper_limit = 75

popularity_data.loc[boxoffice_df['popularity']>popularity_upper_limit] = popularity_upper_limit
budget_data.loc[boxoffice_df['budget']>budget_upper_limit] = budget_upper_limit

# plt.boxplot(popularity_data)
# plt.show()

# print("Number of genres in films:")
# print(boxoffice_df['genres'].apply(lambda x: len(x) if x != {} else 0).value_counts())


#data preparation


processed_df = boxoffice_df.drop(['belongs_to_collection','homepage','overview','poster_path','production_companies','imdb_id',
                                   'spoken_languages','production_countries','tagline','Keywords','cast','crew'],axis=1)

# filling missing atributes

processed_df.runtime.fillna(value=processed_df.runtime.mean(),inplace=True)
print(processed_df.dtypes)
print(processed_df.apply(lambda x: x.isnull().any()))


print(processed_df.head(10))

#strings to categorical features
processed_df['original_language'] = processed_df.original_language.astype('category')
processed_df['original_title'] = processed_df.original_title.astype('category')
processed_df['release_date'] = processed_df.release_date.astype('category')
processed_df['status'] = processed_df.status.astype('category')
processed_df['title'] = processed_df.title.astype('category')
processed_df['all_genres'] = processed_df.all_genres.astype('category')

processed_df['original_language'] = pd.get_dummies(processed_df['original_language'])
processed_df['original_title'] = pd.get_dummies(processed_df['original_title'])
processed_df['release_date'] = pd.get_dummies(processed_df['release_date'])
processed_df['status'] = pd.get_dummies(processed_df['status'])
processed_df['title'] = pd.get_dummies(processed_df['title'])
processed_df['all_genres'] = pd.get_dummies(processed_df['all_genres'])


# separate class from features
data_features = processed_df.drop(['revenue'],axis=1).values
data_labels = processed_df['revenue'].values



#pca  


pca = PCA(n_components=4)
fit = pca.fit(data_features)
print("Explained Variance:")
print(fit.explained_variance_ratio_)
print(fit.components_)

#univariate selection 
# test = SelectKBest(score_func=f_classif,k=4)
# fit = test.fit(data_features,data_labels)
# np.set_printoptions(precision=3)
# print(fit.scores_)
# features = fit.transform(data_features)
# print(features[0:10])

#min_max

scaler = MinMaxScaler()
scaler.fit(data_features)
features = scaler.fit_transform(data_features)

#standard
scaler = StandardScaler()
features = scaler.fit_transform(data_features)


X_train, X_test, Y_train, Y_test = train_test_split(features, data_labels, test_size=0.3)
#decision tree 

clf = DecisionTreeRegressor(max_depth=5,max_features=1)

clf.fit(X_train,Y_train)

score = clf.score(X_test,Y_test)

print(score)

y_pred = clf.predict(X_test)



names = ["Decision Tree Regressor", "MLP Regressor", "Random Forest Regressor", "AdaBoost", "Bagging Regressor","Extra Trees Regressor"]

classifiers = [
    DecisionTreeRegressor(max_depth=5,max_features=1),
    MLPRegressor(alpha=1,max_iter=200,power_t=0.9,batch_size=50),
    RandomForestRegressor(max_depth=5, max_features=1, n_estimators=10),
    AdaBoostRegressor(n_estimators=10),
    BaggingRegressor(max_features=1,n_estimators=10,base_estimator=clf),
    ExtraTreesRegressor(max_depth=5)
]

for name, clf in zip(names, classifiers):
    clf.fit(X_train,Y_train)
    score = clf.score(X_test,Y_test)
    y_pred = clf.predict(X_test)
    print(name+": "+str(score))
    mse = mean_squared_log_error(Y_test,y_pred)
    print('MSE: %.4f' % mse)
    # print(confusion_matrix(Y_test,y_pred,labels=None))
    # print(cohen_kappa_score(Y_test,y_pred, labels=None))
    # print(classification_report(Y_test,y_pred,labels=None))