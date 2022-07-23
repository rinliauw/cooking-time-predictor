# Cooking Time Predictor

## Data Preprocessing Stage
1. Load the data set for the train such as:
```python
x_train_original = pd.read_csv(r"recipe_train.csv", index_col = False, delimiter = ',', header=0)
```
Preprocess all the text feature such as removing the punctuation on all of the text feature before splitting the training data into train and test set to evaluate our model.
```python
a = r"[\"\',\[\]]"
x_train_original['name']=(x_train_original['name']).str.replace(a, "", regex=True)
x_train_original['steps']=(x_train_original['steps']).str.replace(a, "", regex=True)
x_train_original['ingredients']=(x_train_original['ingredients']).str.replace(a, "", regex=True)
```

2. Use Countvectorizer to convert a collection of text and get the word frequency for name, step, ingredients features. stop_words attribute is used as well to remove less important words from the text. A sample for name: 
```python
name=(X_train['name'])
name_test=(X_test['name'])
vectorizer = CountVectorizer(stop_words='english')
vectorizer.fit(name)
names=vectorizer.transform(name)
names_test=vectorizer.transform(name_test)
```

3. Using Chi-square to get the first 1000 best features for names,ingredients, steps. Once we get them,we use hstack to combine all the features back in the train and test set for model evaluation. A sample for name:
```python
x2 = SelectKBest(chi2, k=1000)
X_train_x2_names = x2.fit_transform(names,y_train)
X_test_x2_names = x2.transform(names_test)
all_features_x2=hstack((X_train_x2_names, X_train_x2_step,X_train_x2_ingr,X_train[ 'n_steps'].to_numpy().reshape(-1,1),X_train[ 'n_ingredients'].to_numpy().reshape(-1,1)))
all_features_test_x2=hstack((X_test_x2_names, X_test_x2_step,X_test_x2_ingr,X_test[ 'n_steps'].to_numpy().reshape(-1,1),X_test[ 'n_ingredients'].to_numpy().reshape(-1,1)))
```


## Train and Test Model Stage (using `recipe_train.csv`)
1. Logistic Regression
- To train the Logistic Regression model:
```python
lr_2 = LogisticRegression(max_iter=2000, C=0.5).fit(all_features_x2, y_train)
```

- To get prediction results for Logistic Regression model:

```python
print("Logistic regression accuracy for Chisquare:",lr_2.score(all_features_test_x2,y_test))
```

2. Naive Bayes
- Multinomial NB
To train the Multinomial Naive Bayes model:

```python
Multi = MultinomialNB(alpha=10).fit(all_features_x2.toarray(), y_train)
```
To get prediction results for MNB Model:
```python
print("Multinomial accuracy for Chisquare:", Multi.score(all_features_test_x2.toarray(),y_test))
```

- Bernoulli NB
To train the Bernoulli NB model:
```python
bernoulli= BernoulliNB(alpha=6).fit(all_features_x2.toarray(), y_train)
```

To get prediction results for BNB Model:
```python
print("bernoulli accuracy:",bernoulli.score(all_features_test_x2.toarray(),y_test))
```
- Gaussian NB
To train the Gaussian NB model:
```python
gaus= GaussianNB().fit(all_features_x2.toarray(), y_train)
```
To get prediction results for Gaussian NB Model:
```python
print("Gaussian NB accuracy for Chisquare:",gaus.score(all_features_test_x2.toarray(),y_test))
```

3. Decision Tree Classifier
- To train the Decision Tree model:
```python
dt= DecisionTreeClassifier(max_depth=20).fit(all_features_x2.toarray(), y_train)
```
- To get prediction results for Decision Tree Model:
```python
print("decision tree accuracy:", dt.score(all_features_test_x2.toarray(),y_test))
```
4. Neural Networks (Multilayer Perceptron Classifier)
- To train the MLP Classifier model without Standard Scaler:
```python
clf = MLPClassifier(max_iter=2000)
clf.fit(all_features_x2.toarray(), y_train
```
- To train the MLP Classifier model with Standard Scaler:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(all_features_x2.toarray())
clf = MLPClassifier(max_iter=2000, alpha=0.1)
clf.fit(x_train, y_train)
x_test = scaler.fit_transform(all_features_test_x2.toarray())
```
- To get prediction results for MLP Classifier Model
Without Standardscaler:
```python
print("NN with relu activation function and alpha = 0.0001 and 100 neurons in 1st hidden layer:", clf.score(all_features_test_x2.toarray(),y_test))
```
With Standardscaler:
```python
print("NN with relu activation function and alpha = 0.0001 and 100 neurons in 1st hidden layer:", clf.score(x_test, y_test))
```

5. Ensemble stacking
To train the Ensemble Stacking model:
- Logistic Regression, Decision Tree and Multinomial NB for base model and Logistic Regression for learner model
```python
level0 = list()
level0.append(('lr', LogisticRegression(max_iter=5000, C=0.5)))
level0.append(('dt', DecisionTreeClassifier(max_depth=10)))
level0.append(('Multinomial NB', MultinomialNB(alpha=10)))
level1 = LogisticRegression(C=0.5,max_iter=5000)
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=3)
model.fit(all_features_x2.toarray(), y_train)
```
- To predict the Ensemble Stacking model:
```python
print("stacking method accuracy:",model.score(all_features_test_x2.toarray(),y_test))
```
- Logistic Regression, Decision Tree and Bernoulli NB for base model and Logistic Regression for learner model
```python
level0 = list()
level0.append(('lr', LogisticRegression(max_iter=5000, C=0.5)))
level0.append(('dt', DecisionTreeClassifier(max_depth=10)))
level0.append(('BNB', BernoulliNB(alpha=6)))
level1 = LogisticRegression(C=0.5,max_iter=5000)
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=3)
model.fit(all_features_x2.toarray(), y_train)
- To predict the Ensemble Stacking model:
print("stacking method accuracy:",model.score(all_features_test_x2.toarray(),y_test))
```


## Train and Test Model Stage (using `recipe_train.csv ` and `recipe_test.csv` for Kaggle submission)

- Using all the dataset from recipe_train.csv instead of train splitting , we do the same step such as Preprocess the data, Feature Engineering, Feature Selection and finally, train the model 

- Predicting test set and make a new csv that contains all the predicted label using LogisticRegression
```python
lr = LogisticRegression(max_iter=2000).fit(all_features_train, label)
pred_test=lr.predict(all_features_test)
d2 = { 'duration_label':pred_test}
df2=pd.DataFrame(d2)
df2.index+=1
index=df2.index
index.name = "id"
df2.to_csv ('logistic_prediction.csv')
```

- Predicting test set and make a new csv that contains all the predicted label using LogisticRegression and tune the parameter to 0.5
```python
lr_2 = LogisticRegression(max_iter=2000, C=0.5).fit(all_features_train, label)
pred_test=lr_2.predict(all_features_test)
d2 = { 'duration_label':pred_test}
df2=pd.DataFrame(d2)
df2.index+=1
index=df2.index
index.name = "id"
df2.to_csv ('logistic_tune_paramter_prediction.csv')
```

- Predicting test set and make a new csv that contains all the predicted label using MultinonialNB
```python
MNB = MultinomialNB(alpha=10).fit(all_features_train, label)
pred_test=MNB.predict(all_features_test)
d3 = { 'duration_label':pred_test}
df3=pd.DataFrame(d3)
df3.index+=1
index=df3.index
index.name = "id"
df3.to_csv ('MultinomialNB_prediction.csv')
```

- Predicting test set and make a new csv that contains all the predicted label using BernoulliNB
BNB = BernoulliNB(alpha=6).fit(all_features_train.toarray(), label)
```python
pred_test=BNB.predict(all_features_test.toarray())
d4 = { 'duration_label':pred_test}
df4=pd.DataFrame(d4)
df4.index+=1
index=df4.index
index.name = "id"
df4.to_csv('BernoulliNB_prediction.csv')
```
- Predicting test set and make a new csv that contains all the predicted label using GaussianNB
```python
GNB = GaussianNB().fit(all_features_train.toarray(), label)
pred_test=GNB.predict(all_features_test.toarray())
d5 = { 'duration_label':pred_test}
df5=pd.DataFrame(d5)
df5.index+=1
index=df5.index
index.name = "id"
df5.to_csv ('GaussianNB_prediction.csv')
```
- Predicting test set and make a new csv that contains all the predicted label using DecisionTree
```python
dt = DecisionTreeClassifier(max_depth=20).fit(all_features_train, label)
pred_test=dt.predict(all_features_test)
d6 = { 'duration_label':pred_test}
df6=pd.DataFrame(d6)
df6.index+=1
index=df6.index
index.name = "id"
df6.to_csv ('Decison_tree_prediction.csv')
```
- Predicting test set and make a new csv that contains all the predicted label using Neural Network
```python
clf = MLPClassifier(max_iter=2000)
clf.fit(all_features_train.toarray(), label)
pred_test=clf.predict(all_features_test)
d7 = { 'duration_label':pred_test}
df7=pd.DataFrame(d7)
df7.index+=1
index=df7.index
index.name = "id"
df7.to_csv ('Neural_Network_prediction.csv')
```

- Predicting test set and make a new csv that contains all the predicted label using ensemble Stacking
```python
level0 = list()
level0.append(('lr', LogisticRegression(max_iter=2000)))
level0.append(('dt', DecisionTreeClassifier(max_depth=20)))
level0.append(('MNB', MultinomialNB()))
level1 = LogisticRegression(C=1,max_iter=2000)
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
model.fit(all_features_train.toarray(), label)
y_pred_test=model.predict(all_features_test.toarray())
d1 = { 'duration_label':y_pred_test}
df1=pd.DataFrame(d1)
df1.index+=1
index=df1.index
index.name = "id"
df1.to_csv ('kaggle_dataframe.csv')
```