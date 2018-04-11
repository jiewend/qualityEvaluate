import pandas as pd
import numpy as np
import my_module 

df = pd.DataFrame({'a':[1, 2, 3, 4, 5, 6],
    'b':[0, 0, 0, 0, 0 ,0],
    'c':[6, 6, 78, 5, 78, 8],
    'd':[1, 1, 1, 1, 1, 1]})
df1 = pd.DataFrame({'e':[1, 2, 3, 4, 5, 6],
    'f':[0, 0, 0, 0, 0 ,0],
    'g':[6, 6, 78, 5, 78, 8],
    'h':[1, 1, 1, 1, 1, 1]})

df_list = [df, df1]

num_zero = my_module.num_std_is_zero(df_list)
print(num_zero)

df_list = my_module.del_std_is_zero_list(df_list)

num_zero = my_module.num_std_is_zero(df_list)
print(num_zero)
