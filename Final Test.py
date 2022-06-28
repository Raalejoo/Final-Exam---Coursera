import itertools
import numpy as np
import matplotlib.pyplot as plt
from jinja2.filters import K
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker
from sklearn import preprocessing
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.linear_model import LogisticRegression


''' Importación de Data'''


file = 'C:\\Users\\rarraiz\\Desktop\\Machine Learning\\Insumos\\loan_train.csv'
df = pd.read_csv(file, sep=',')

file = 'C:\\Users\\rarraiz\\Desktop\\Machine Learning\\Insumos\\loan_test.csv'
df_test = pd.read_csv(file, sep=',')

'''Tamaño de Data'''

df.shape

'''Convertir Fechas en Formatos de Fechas'''

df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])

'''Veamos cuántos de cada clase hay en nuestro conjunto de datos'''

df['loan_status'].value_counts()

'''Por Principal'''

prin = np.linspace(df.Principal.min(), df.Principal.max(), 10)
parameter = sns.FacetGrid(df, col='Gender', hue='loan_status', palette='Set1',
                          col_wrap=2)
parameter.map(plt.hist, 'Principal', bins=prin, ec='k')

parameter.axes[-1].legend()
plt.show()


'''Por Edades'''

age = np.linspace(df.age.min(), df.age.max(), 10)
parameter = sns.FacetGrid(df, col='Gender', hue='loan_status', palette='Set1', col_wrap=2)
parameter.map(plt.hist, 'age', bins=age, ec='k')

parameter.axes[-1].legend()
plt.show()

'''Preprocesamiento: Selección/extracción de características'''

df = df.assign(DayofWeek=df['effective_date'].dt.dayofweek)

day = np.linspace(df.DayofWeek.min(), df.DayofWeek.max(), 10)
parameter = sns.FacetGrid(df, col='Gender', hue="loan_status", palette='Set1', col_wrap=2)
parameter.map(plt.hist, 'DayofWeek', bins=day, ec='k')

parameter.axes[-1].legend()
plt.show()

'''Obteniendo la semana'''

df = df.assign(Weekend=df['DayofWeek'].apply(lambda x: 1 if(x>3) else 0))

df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)

'''Convertir características categóricas en valores numéricos'''

df['Gender'] = df['Gender'].replace(['male', 'female'], [0, 1])
df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)

'''One Hot Encoding'''

df1 = df.reindex(columns=['Principal', 'terms', 'age', 'Gender', 'education'])

data = df.reindex(columns=['Principal', 'terms', 'age', 'Gender', 'Weekend'])
data = pd.concat([data, pd.get_dummies(df['education'])], axis=1).drop(['Master or Above'], axis=1)
data.head()


X = data.copy()
y = df['loan_status'].values

'''Normalizando Data'''

X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

'''Clasificación'''

'''K Nearest Neighbor(KNN)'''

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
X_train


'''Creando k Vecinos'''

Ks = 10
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))
ConfustionMx = [];
for i in range(1, Ks):

    neighbors = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
    yhat = neighbors.predict(X_test)
    mean_acc[i - 1] = metrics.accuracy_score(y_test, yhat)

    std_acc[i - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

mean_acc

'''Crendo Grafica'''

plt.plot(range(1, Ks), mean_acc, 'g')
plt.fill_between(range(1, Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Precisión ', '+/- 3xstd'))
plt.ylabel('Precisión')
plt.xlabel('Numero de Vecinos (K)')
plt.tight_layout()
plt.show()

print('La Mejor Precisión fue con ', mean_acc.max(), 'con K= ', mean_acc.argmax() + 1)


'''Árbol de Decisiones'''

Tree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
Tree.fit(X_train,y_train)
predTree = Tree.predict(X_test)

tree.plot_tree(Tree)
plt.show()

'''Support Vector Machine'''

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
yhatsvm = clf.predict(X_test)
yhatsvm [0:5]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cnf_matrix = confusion_matrix(y_test, yhat)
np.set_printoptions(precision=2)

print(classification_report(y_test, yhat))

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=df['loan_status'].unique().tolist(), normalize=False,
                      title='Confusion matrix')

plt.show()

'''Logistic Regression'''

LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR

yhatl = LR.predict(X_test)
yhatl

yhat_prob = LR.predict_proba(X_test)


'''Conjunto de pruebas para la evaluación'''

df_test['Gender'].replace(to_replace=['male','female'], value=[np.int64(0),np.int64(1)],inplace=True)

df_test.groupby(['education'])['loan_status'].value_counts(normalize=True)
Feature = df_test[['Principal','terms','age','Gender']]
Feature = pd.concat([Feature,pd.get_dummies(df_test['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()
X2 = Feature
y2 = df_test['loan_status'].values
X2= preprocessing.StandardScaler().fit(X2).transform(X2)



'''Reporte'''

'Arbol'
predTree = Tree.predict(X_test)
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))

'KNeighbors'
cr = classification_report(y_test, yhat)
print(cr)

'SVM'
classification_report(y_test,yhatsvm )
print('SVM ACCY:', classification_report(y_test, yhatsvm))

'Logistic'
classification_report(y_test, yhatl)
print('Logit', classification_report(y_test, yhatl))
