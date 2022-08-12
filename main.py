import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from os import listdir
from detector_barridos import detector_barrido
from detector_barridos import wdw_edges
from convert_to_rows import convert_to_rows
from swept_wdw_twp import swept_window_two

freq_dir= listdir(r'C:\Users\Odón\PycharmProjects\dataset_2_V2\freq_data')
power_dir= listdir(r'C:\Users\Odón\PycharmProjects\dataset_2_V2\power_data')

window= 0.8
power_rx= []
aux= 0
freq_files= r'C:\Users\Odón\PycharmProjects\dataset_2_V2\freq_data'
power_files= r'C:\Users\Odón\PycharmProjects\dataset_2_V2\power_data'
slash='\\'

#Leemos todos los archivos de medidas y obtenemos los barridos
for i in freq_dir:
    power_rx.append(detector_barrido(freq_files+slash+i, power_files+slash+power_dir[aux], window))
    aux += 1

df = pd.DataFrame(power_rx)
tag= pd.read_excel('Tags.xlsx')

df = pd.concat([tag, df], axis=1)

#df es un pandas dataframe, una columna indica a que tag pertenecen las medidas
# las siguientes columnas de una misma fila indican los diferentes barridos dentro de
#una misma medida. Es decir una fila entera corresponde a una medida de la Ettus

print(df)

#df.to_excel('FINALdata.xlsx')

cont1= 0
cont2= 0
freq_edges= wdw_edges(1.5, 5, window)

#df.to_excel('data3.xlsx')

'''
while cont1 <= 0:
    while cont2 <= 0:
        if (df.loc[cont1].at[cont2])!=None and df.loc[cont1].at['Tag']==1:
            plt.plot(np.linspace(freq_edges[0], freq_edges[1], num= len(df.loc[cont1].at[cont2])), df.loc[cont1].at[cont2])
        cont2 += 1
    cont1 += 1
    cont2 = 0

plt.show()



tag_1 = df[df["Tag"] == 1]
tag_2 = df[df["Tag"] == 2]
tag_3 = df[df["Tag"] == 3]
tag_4 = df[df["Tag"] == 4]


tag_1.to_excel('tag_1.xlsx')
tag_2.to_excel('tag_2.xlsx')
tag_3.to_excel('tag_3.xlsx')
tag_4.to_excel('tag_4.xlsx')
'''

dataset, barridos_list= convert_to_rows(df)
df_tags=dataset['Tag']

#print(dataset)
#dataset.to_excel('FINALdataset.xlsx')
#print("YA")

#Este dataset es un dataframe de dos columnas una indica el tag
#y otra es un barrido de este



#Codigo para extaer numero de muestras no nulas de un barrido
#ejecutado una vez y exportado resultados a excel pq tarda
#mucho en ejecutar

'''
(rows, columns)= dataset.shape

cont1= 0
cont2= 4000
under_data= np.zeros(rows)


while cont1 <= rows - 1:
    while cont2 <= columns - 2:
        if (np.isnan(dataset.loc[cont1].at[cont2])):
            under_data[cont1]= cont2
            break


        cont2 += 1
    cont1 += 1
    print(cont1)
    cont2 = 4000


cont1= 0
cont2= columns-2
while cont1 <= rows - 1:
    while cont2 >= 0:
        if (np.isnan(dataset.loc[cont1].at[cont2])):
            pass
        else:
            under_data[cont1] = cont2 ||||(aqui habria que poner un +1)||||
            break


        cont2 -= 1
    cont1 += 1
    print(cont1)
    cont2 = columns-2

df_under = pd.DataFrame(under_data)
df_under.to_excel('FINALunder_data.xlsx')

'''

df_under= pd.read_excel('FINALunder_data.xlsx')

r = len(df_under.axes[0])

indices= df_under[df_under.num_muestras < 4479].index
df_under = df_under.drop(indices)

for i in indices:
    df_tags = df_tags.drop(i)

list_aux= df_tags.values.tolist()
df_tags = pd.DataFrame(list_aux,columns =['Tag'])

# df_under = df_under.drop(df_under[df_under.num_muestras < 4000].index)

#data_len= df_under["num_muestras"].min()
data_len= 4479
# Make all data same length


cont1= 0

while cont1<= r-1:
    if cont1 in indices:
        pass
    else:
        if cont1==0:
            barridos_list_minLen= [swept_window_two(df_under.loc[cont1].at["num_muestras"], data_len, barridos_list[cont1])]
        else:
            barridos_list_minLen.append(swept_window_two(df_under.loc[cont1].at["num_muestras"], data_len, barridos_list[cont1]))
    cont1 += 1

df_barridos = pd.DataFrame(barridos_list_minLen)
df_barridos.to_excel('FINALrecortado.xlsx')
df_barridos = pd.concat([df_tags, df_barridos], axis=1)
print(df_barridos)

df_barridos.to_excel('FINALrecortado2.xlsx')
#dataset.to_excel('NEWdataset.xlsx')

