'''Question 1 Part 1'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('/Users/hanimoudarres/Downloads/Foundations of ML/HW 2/wrangling/assignment/data/airbnb_hw.csv',low_memory=False)

var = 'Price'

# Initial description before coercion
print('Before coercion: \n', df[var].describe(),'\n')

# Coerce to numeric and create missing indicator
df[var] = pd.to_numeric(df[var], errors='coerce')
df["Price_nan"] = df[var].isnull()

# Description after coercion
print('After coercion: \n', df[var].describe(),'\n')

# Total missing values: 181
print('Total Missings: \n', sum(df['Price_nan']),'\n')

# Replace spaces with NaN
df[var] = df[var].replace(' ',np.nan)

# Plotting the distribution
plt.figure(figsize=(10,6))
plt.hist(df[var].dropna(), bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Price after Coercion', fontsize=16)
plt.xlabel('Price', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(axis='y', alpha=0.75)
plt.show()


'''Question 1 Part 2'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/hanimoudarres/Downloads/Foundations of ML/HW 2/wrangling/assignment/data/mn_police_use_of_force.csv',low_memory=False)

var = 'subject_injury'

# Initial unique values
print(df[var].unique(), '\n')

# Create missing indicator
df['empty_subject_injury'] = df[var].isnull()

# Replace empty strings with NaN
df[var] = df[var].replace('', np.nan)

# Value counts after replacement
print(df[var].value_counts(), '\n') 

# Total missing values
# The vast majority of entries are missing!!
print('Total Missings: \n', sum(df['empty_subject_injury']), '\n')

# Cross Tabulation
print(pd.crosstab(df['subject_injury'], df['force_type']))
print (pd.crosstab(df['empty_subject_injury'], df['force_type']))
print('\n\n')




"""Question 1 Part 3"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet('/Users/hanimoudarres/Downloads/Foundations of ML/HW 2/wrangling/assignment/data/justice_data.parquet', engine='pyarrow')

var = 'WhetherDefendantWasReleasedPretrial'

# Initial unique values
# 0 = false, 1 = true, 9 = missing
print(df[var].unique(), '\n')

# Create missing indicator
df['empty_release_pretrial'] = df[var] == 9

# Replace empty strings with NaN
df[var] = df[var].replace(9, np.nan)

# Value counts after replacement
print(df[var].value_counts(), '\n') 

# Total missing values
print('Total Missings: \n', sum(df['empty_release_pretrial']), '\n')


"""Question 1 Part 4"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/hanimoudarres/Downloads/Foundations of ML/HW 2/wrangling/assignment/data/mn_police_use_of_force.csv',low_memory=False)

var = 'ImposedSentenceAllChargeInContactEvent'

