import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def convert_to_rows(df):

    (rows, columns)= df.shape
    cont1= 0
    cont2= 0
    cont3= 0

    while cont1 <= rows-1:
        while cont2 <= columns-2:
            if (df.loc[cont1].at[cont2]) != None:

                if cont3==0:
                    tag_list = [df.loc[cont1].at['Tag']]
                    barridos_list = [df.loc[cont1].at[cont2]]
                    cont3= 1
                else:
                    tag_list.append(df.loc[cont1].at['Tag'])
                    barridos_list.append(df.loc[cont1].at[cont2])
            cont2 += 1
        cont1 += 1
        cont2 = 0

    df_tags = pd.DataFrame(tag_list,columns =['Tag'])
    df_barridos = pd.DataFrame(barridos_list)
    df2 = pd.concat([df_tags, df_barridos], axis=1)

    return df2, barridos_list
