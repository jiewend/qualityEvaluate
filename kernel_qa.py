import pandas as pd
import numpy as np
from matplotlib.pyplot import plot, ion, show

def is_tool_name(X_ser):
    return
def std_Ser_or_Df(X, isdf = False):
    if isdf :
        X_df = X
        X_df_std = pd.DataFrame()
        for i in range(X_df.shape[1]):
            if X_df.iloc[:, i].dtype = 'object':
                X_df[X_df.columns[i]] = (X_df.iloc[:, i] - X_df.iloc[:, i].mean()) / X_df.iloc[:, i].std()
        X = X_df
    else :
        X = (X - X.mean()) / X.std()
    return X


    
def df_list_to_df(df_list):
    df = pd.DataFrame()
    for i in range(len(df_list)):
        df[df_list[i].columns] = df_list[i]
    return df

def corr_with_y(X_df, y):
    y_Series = y.iloc[:, 0]
    corr_Series = pd.Series()
    for i in range(X_df.shape[1]):
        if X_df.iloc[:, i].dtype == 'float':
            corr_Series[X_df.columns[i]] = y_Series.corr(X_df.iloc[:, i])
    return corr_Series


def num_nan_df_list(df_list):
    num_nan = pd.Series()
    for i in range(len(df_list)):
        for j in range(df_list[i].shape[1]):
            index = df_list[i].columns[j]
            num_nan[index] = sum(pd.isna(df_list[i].iloc[:, j]))
    return num_nan

def num_nan_df(df):
    num_nan = pd.Series()
    for j in range(df.shape[1]):
        index = df.columns[j]
        num_nan[index] = sum(pd.isna(df.iloc[:, j]))
    return num_nan

def num_std_is_zero(df_list) :
    count = 0
    count_shape = 0
    for i in range(len(df_list)):
        count = count + sum(df_list[i].std() == 0)
        count_shape = count_shape + df_list[i].shape[1]
        print('total : %d  useless : %d' %(df_list[i].shape[1], sum(df_list[i].std() == 0)))
    print('total : %d  uselsee : %d ---------------' %(count, count_shape))
    return count

def del_nan_columns(df, n):
    del_columns = []
    for i in range(df.shape[1]):
        if sum(pd.isna(df.iloc[:, i])) > n:
            del_columns.append(df.columns[i])
    df.drop(del_columns, axis=1, inplace=True)        
    return

def del_std_is_zero(df):
    zero_colum = ((df.std() == 0)[(df.std() == 0) == True]).index
    df.drop(zero_colum, axis=1, inplace=True)
    return df

def del_std_is_zero_list(df_list) :
    for i in range(len(df_list)):
        del_std_is_zero(df_list[i])  
    return df_list 



#  train  = pd.read_excel('./data/train.xlsx')
#  sub_process_TOOL_ID = train.iloc[:, 1:232]
#  sub_process_Tool = train.iloc[:, 232:752]
#  sub_process_TOOL_ID1 = train.iloc[:, 752:774]
#  sub_process_TOOL_ID2 = train.iloc[:, 774:945]
#  sub_process_TOOL_ID3 = train.iloc[:, 945:1702]
#  sub_process_Tool1 = train.iloc[:, 1702:2382]
#  sub_process_Tool2 = train.iloc[:, 2382:3694]
#  sub_process_tool = train.iloc[:, 3694:3894]
#  sub_process_tool1 = train.iloc[:, 3894:4169]
#  sub_process_TOOL = train.iloc[:, 4169:5978]
#  sub_process_TOOL1 = train.iloc[:, 5978:6192]
#  sub_process_TOOL2 = train.iloc[:, 6192:6575]
#  sub_process_TOOL3 = train.iloc[:, 6575:8028]
#  y = train.iloc[:, 8028:8029]

#  sub_process_sets = [sub_process_TOOL_ID, sub_process_TOOL_ID1
    #  , sub_process_TOOL_ID2, sub_process_TOOL_ID3
    #  , sub_process_Tool, sub_process_Tool1, sub_process_Tool2
    #  , sub_process_tool, sub_process_tool1
    #  , sub_process_TOOL, sub_process_TOOL1, sub_process_TOOL2, sub_process_TOOL3]

#  backup_sets = sub_process_sets
#  del train

#  num_zero = num_std_is_zero(sub_process_sets)

#  del_std_is_zero_list(sub_process_sets)

#  num_zero1 = num_std_is_zero(sub_process_sets)

#  num_nan = num_nan(sub_process_sets)

#  X_df = df_list_to_df(sub_process_sets)


# load data without std == 0
X_df = pd.read_csv('X_df.csv')
y = pd.read_csv('y.csv')
# 7078 colums lefta,done with the std == 0
num_nan = num_nan_df(X_df)

del_nan_columns(X_df, 100)

y_std = std_Ser_or_Df(y)
X_df_std = std_Ser_or_Df(X_df, isdf=True)
corr_Series = corr_with_y(X_df_std, y_std)
# 6933 columns left, done with the nan > 100















