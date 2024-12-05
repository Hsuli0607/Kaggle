# %%
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier

# %%
### load file 
data = pd.read_csv('train.csv')
# %%
### Working on missing data and filter unnessary features 
data_no_missing = data.dropna(subset=['Age','Embarked']).drop(['Cabin',
                                                               'Name','Ticket'], axis=1)
# %%
### maping "Sex" to 0, 1
data_no_missing['Sex'] = data_no_missing['Sex'].map({'male': 0, 'female': 1}
                                                    ).astype(int)
### Covert 'Embarked' to C = Cherbourg :0, Q = Queenstown:1, S = Southampton:2
data_no_missing['Embarked'] = data_no_missing['Embarked'].map(
    {'C':0, 'Q':1, 'S':2}).astype(int)
# %%
### add one new feature "Along" (total person number)
data_no_missing['Along'] = (data_no_missing['SibSp'] + data_no_missing['Parch']
                            ).apply(lambda x: 1 if x==0 else 0)
# %% 
### correlation matrix
def plot_correlation_matrix(data):
    correlation = data.corr()
    print(correlation)

    fig, ax = plt.subplots(1, 1)
    cax = ax.matshow(correlation, vmin=-1, vmax=1, interpolation='none')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(correlation.columns)))
    ax.set_yticks(np.arange(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=90)
    ax.set_yticklabels(correlation.columns)
    plt.show()
plot_correlation_matrix(data_no_missing)
# %%
### Visulaize important features
# Plot bar chart of Sex vs Survived
data_no_missing[['Sex', 'Survived']].groupby('Sex').mean().plot(kind='bar')
plt.title('Survival Rate by Sex')
plt.xlabel('Sex (0 = Male, 1 = Female)')
plt.ylabel('Survival Rate')
plt.show()

# Plot bar chart of Pclass vs Survived
data_no_missing[['Pclass', 'Survived']].groupby('Pclass').mean().plot(kind='bar')
plt.title('Survival Rate by Pclass')
plt.xlabel('Pclass (1 = 1st, 2 = 2nd, 3 = 3rd)')
plt.ylabel('Survival Rate')
plt.show()

# Plot bar chart of Along vs Survived
data_no_missing[['Along', 'Survived']].groupby('Along').mean().plot(kind='bar')
plt.title('Survival Rate by Along')
plt.xlabel('Along (1 = Along, 0 = With Family)')
plt.ylabel('Survival Rate')
plt.show()

# %%
### filter unnecessary features again and hold out train_X, train_Y 
train_Y = data_no_missing['Survived'].values.flatten()
# Separate features and target variable
train_X = data_no_missing.drop(['PassengerId', 'Survived', 'SibSp', 'Parch'], axis=1).values



# %%
### folds, seed, scoring setting 
num_folds = 10
seed = 7
scoring = 'accuracy'

# %%
### Spot-check Algorithms baseline
models = []
models.append(('LR', LogisticRegression(solver='newton-cholesky')))
models.append(('LD', LinearDiscriminantAnalysis()))
models.append(('QDR', QuadraticDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

def evaluate_models(models, train_X, train_Y, num_folds, seed, scoring):
    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        cv_results = cross_val_score(model, train_X, train_Y, scoring=scoring, cv=kfold)
        results.append(cv_results)
        names.append(name)
        print(f"{name}, {cv_results.mean(), cv_results.std()}")

    # Boxplot for Algorithms Comparison
    fig, ax = plt.subplots(1, 1)
    fig.suptitle = 'Algorithms Comparison'
    plt.boxplot(results)
    ax.set_xticklabels(names, rotation=-45)
    plt.show()
evaluate_models(models, train_X, train_Y, num_folds, seed, scoring)
# %%
### Standardized data Algorithm spot-check
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()), 
                                        ('LR', LogisticRegression())])))
pipelines.append(('ScaledLD', Pipeline([('Scaler', StandardScaler()), 
                                        ('LD', LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledQDR', Pipeline([('Scaler', StandardScaler()), 
                                        ('QDR', QuadraticDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()), 
                                        ('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()), 
                                        ('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()), 
                                        ('NB', GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()), 
                                        ('SVM', SVC())])))

evaluate_models(pipelines, train_X, train_Y, num_folds, seed, scoring)

# %%
### SVM Tuning with pipeline
def model_tuning(scaler, algorithm, train_X, train_Y,param_grid,scoring='accuracy'
                 ,num_folds=10,seed=7):
    model = Pipeline([('scaler', scaler), ('algorithm', algorithm)])
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(train_X, train_Y)
    print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
            print(f"({mean}, {stdev}, {param})")


scaler = StandardScaler()
algorithm = SVC()
c_values = np.arange(0.1, 2.0, 0.2)
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(algorithm__C=c_values, algorithm__kernel=kernel_values)

model_tuning(scaler, algorithm, train_X, train_Y,param_grid)
# %%
### QDR Tuning
scaler = StandardScaler()
algorithm = QuadraticDiscriminantAnalysis()
reg_param = np.arange(0, 1.0, 0.1)
param_grid = dict(algorithm__reg_param=reg_param)
model_tuning(scaler, algorithm, train_X, train_Y,param_grid)


# %%
### Ensemble Methods in Pipeline
pipelines = []
pipelines.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()), 
                                        ('AB', AdaBoostClassifier(algorithm='SAMME'))])))
pipelines.append(('ScaledGB', Pipeline([('Scaler', StandardScaler()), 
                                        ('GB', GradientBoostingClassifier())])))
pipelines.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()), 
                                        ('RF', RandomForestClassifier())])))
pipelines.append(('ScaledET', Pipeline([('Scaler', StandardScaler()), 
                                        ('ET', ExtraTreesClassifier())])))
    
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, train_X, train_Y, scoring=scoring, cv=kfold)
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean()}, {cv_results.std()}")
    
    # Boxplot for Ensemble Methods Comparison
fig, ax = plt.subplots(1,1)
fig.suptitle('Ensemble Methods Comparison')
plt.boxplot(results)
ax.set_xticklabels(names, rotation=-45)
plt.show()

# %% 
### GB Tuning
scaler = StandardScaler()
algorithm = GradientBoostingClassifier()
n_estimators = [20, 30, 40, 50, 100, 150]
learning_rate = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
param_grid = dict(algorithm__n_estimators=n_estimators, algorithm__learning_rate=learning_rate)

model_tuning(scaler, algorithm, train_X, train_Y,param_grid)

# %%
### Test the Model
# load test file

test = pd.read_csv('test.csv')

### maping "Sex" to 0, 1
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1}).astype(int)
### Covert 'Embarked' to C = Cherbourg :0, Q = Queenstown:1, S = Southampton:2
test['Embarked'] = test['Embarked'].map({'C':0, 'Q':1, 'S':2}).astype(int)
### Handle missing values in 'Age' by filling with the median value
test['Age'] = test['Age'].fillna(test['Age'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

### add one new feature "Along" (total person number)
test['Along'] = (test['SibSp'] + test['Parch']).apply(lambda x: 1 if x==0 else 0)

# %%
# Removed features
Removed_features = ['PassengerId', 'SibSp', 'Parch', 'Name', 'Cabin', 'Ticket']
test_X = test.drop(Removed_features, axis=1).values

# %%
### get prediction
scaler = StandardScaler()
rescaled_train_X = scaler.fit_transform(train_X)
model = GradientBoostingClassifier()
model.fit(rescaled_train_X, train_Y)
rescaled_test_X = scaler.transform(test_X)
prediction = model.predict(rescaled_test_X)

# %%
### print the submission.csv
output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': prediction})
output.to_csv('submission.csv', index=False)
print("Predictions saved to submission.csv")















# %%
