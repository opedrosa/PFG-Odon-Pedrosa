from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# features= list(range(0,4478))
x= pd.read_excel('recortado2.xlsx')
y = x.loc[:,['Tag']].values
x = x.drop('Unnamed: 0', 1)
x = x.drop('Tag', 1)

df2= pd.read_excel('FINALrecortado2.xlsx')
y2 = df2.loc[:,['Tag']].values
df2 = df2.drop('Unnamed: 0', 1)
df2 = df2.drop('Tag', 1)

pca = PCA(n_components=20)
pca.fit(x)
principalComponents= pca.transform(df2)
principalDf = pd.DataFrame(principalComponents)
y = pd.DataFrame(data= y2, columns = ['Tag'])
finalDf = pd.concat([principalDf, y], axis = 1)

finalDf.to_excel('FINALreducida20.xlsx')
