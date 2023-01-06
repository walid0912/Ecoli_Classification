# Ecoli Classification
Classification SVM de Ecoli dataset 

# Ecoli Dataset 
Cet ensemble de données contient 336 lignes de données sur E. coli et leur séquence de protéines.

# Technologies 
| Tech | Description |
| --- | --- |
| Langage  | Python  | 
| Librairie  |   keras, seaborn, numpy, pandas,matplotlib    | 
| Kernel SVM     |   RBF, Linear, Polynomial     | 

# Résultats 
|    Metric          | precision  |  recall | f1-score   |support|
| --- | --- | --- | --- | --- |
|          cp  |     0.97 |     0.97     | 0.97 |       37 |
|          im   |    0.65  |    1.00     | 0.79  |      17 |
|         imU    |   0.00   |   0.00     | 0.00    |    10 |
|          om     |  1.00    |  1.00     | 1.00    |     4 | 
|          pp|       0.94     | 1.00    |  0.97    |    16 |
|    accuracy  |              |        |   0.87    |    84 |
|   macro avg  |     0.71     | 0.79  |    0.75    |    84 | 
|weighted avg   |    0.79     | 0.87 |     0.82    |    84 |
