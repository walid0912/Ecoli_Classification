import matplotlib.pyplot as plt #pour plotter les graphes 
from sklearn import svm, datasets #bibliothèque sklearn du machine learning
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#loading the ecoli dataset
ecoli_df = pd.read_csv("ecoli.csv",header=None,sep="\s+")
#on ajoute les noms de colonnes pour pouvoir effectuer le traitement sur le dataset
col_names = ["squence_name","mcg","gvh","lip","chg","aac","alm1","alm2","site"]
ecoli_df.columns = col_names

#info sur le dataset 
ecoli_df.info()

#description du dataset 
ecoli_df.describe()

#affichage des 5 premiers lignes du dataset
ecoli_df.head(5)

#affichage des 5 dernières lignes du dataset
ecoli_df.tail(5)

ecoli_df.isnull().sum()
ecoli_df.loc[:,ecoli_df.dtypes == "object"].columns.tolist()

#classification SVM du nouveau dataset
X = ecoli_df.drop(['squence_name','site'], axis = 1)
Y = ecoli_df['site']
#partionner le dataset en deux 25% pour les tests et 75% pour le training 
X_train4, X_test4, Y_train4, Y_test4 = train_test_split(X, Y, test_size=0.25)
svm = SVC()
clf = svm.fit(X_train4,Y_train4)
pred = svm.predict(X_test4)
print("Precision du model:")
svm.score(X_test4,Y_test4)

#Test de perfromances
from sklearn.metrics import classification_report
print(classification_report(Y_test4, pred))