import pandas as pd
import numpy as np

def num_nan(df_list):
    num_nan = []
    for i in range(len(df_list)):
        temp = []
        for j in range(len(df_list[i].shape[1])):
            temp.append(sum(pd.isna(df_list[i].iloc[:, j])))
        num_nan.append(temp)
    return num_nan

def hello() :
    print('hello')
    return

def num_std_is_zero(df_list) :
    count = 0
    count_shape = 0
    for i in range(len(df_list)):
        count = count + sum(df_list[i].std() == 0)
        count_shape = count_shape + df_list[i].shape[1]
        print('total : %d  useless : %d', %(df_list[i].shape[1], sum(df_list[i].std() == 0)))
    print('total : %d  uselsee : %d' %(count, count_shape))
    return count


def del_std_is_zero(df) :
    zero_colum = df.columns[df.std() == 0]
    df.drop(zero_colum, axis=1, inplace=True)
    return df

def del_std_is_zero_list(df_list) :
    for i in range(len(df_list)):
        del_std_is_zero(df_list[i])  
    return df_list 

#  def num_column_lis_df(df_list):
    #  count = 0
    #  for i in range(len(df_list)):
        #  count = count + df_list[i].shape[1]
        #  print(df_list[i].shape[1])
    #  print('total : %d', count) 
    #  return

        


    
if __name__ == '__main__'    :
    hello()



    
    
