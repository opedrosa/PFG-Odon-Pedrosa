from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

# features= list(range(0,4478))

df= pd.read_excel('FINALrecortado.xlsx')
df2= pd.read_excel('FINALrecortado2.xlsx')
y = df2.loc[:,['Tag']].values
df = StandardScaler().fit_transform(df)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df)
principalDf = pd.DataFrame(data = principalComponents , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df2[['Tag']]], axis = 1)

print(finalDf)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlim([-100,100])
ax.set_ylim([-100,100])
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
tags = [1, 2, 3, 4]
colors = ['r', 'g', 'b', 'y']


for target, color in zip(tags,colors):
    indicesToKeep = finalDf['Tag'] == target
    print(indicesToKeep)
    # print(finalDf.loc[indicesToKeep, 'principal component 1'])
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(tags)
ax.grid()

plt.show()
