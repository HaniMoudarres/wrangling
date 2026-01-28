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

df = pd.read_parquet('/Users/hanimoudarres/Downloads/Foundations of ML/HW 2/wrangling/assignment/data/justice_data.parquet', engine='pyarrow')

var = 'ImposedSentenceAllChargeInContactEvent'

# Initial unique values
# Shows mixed type numerical data that is very messy
print(df[var].unique(), '\n')

# print value counts
print(df[var].value_counts(dropna=False), '\n')

# Replace empty strings with NaN
df[var] = df[var].replace('', np.nan)

# Coerce to numeric
df[var] = pd.to_numeric(df[var], errors='coerce')

# Description after coercion
print(df[var].describe(), '\n')
print(df[var].value_counts().head(10), '\n')

# min and max values
print(df[var].min(), '\n')
print(df[var].max(), '\n')

# Plotting the distribution
# Not a very good distribution due to many extreme outliers
plt.figure(figsize=(10,6))
plt.hist(df[var].dropna(), bins=50, color='lightgreen', edgecolor='black')
plt.title('Distribution of Imposed Sentence After Coercion', fontsize=16)
plt.xlabel('Imposed Sentence (Days)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(axis='y', alpha=0.75)
plt.show()

# Log Transformation
df[var + '_log'] = np.log1p(df[var])

# Plotting the log-transformed distribution
# Much better distribution after log transformation
plt.figure(figsize=(10,6))
plt.hist(df[var + '_log'].dropna(), bins=50, color='salmon', edgecolor='black')
plt.title('Log-Transformed Distribution of Imposed Sentence', fontsize=16)
plt.xlabel('Log(Imposed Sentence + 1)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(axis='y', alpha=0.75)
plt.show()

var2 = 'SentenceTypeAllChargesAtConvictionInContactEvent'

# Initial description
# 0 = no sentence, 1 = jail/short-term, 2 = probation/community supervision, 4 = prison/long-term, 9 = missing
print(df[var2].unique(), '\n')

# Value counts
print(df[var2].value_counts(dropna=False), '\n')

# Replace 9 with NaN
df[var2] = df[var2].replace(9, np.nan)

# Convert to categorical
df[var2] = df[var2].astype('category')

df[var2] = df[var2].cat.rename_categories({
    0: 'No Sentence',
    1: 'Jail/Short-term',
    2: 'Probation/Community Supervision',
    4: 'Prison/Long-term'
})

# Value counts after replacement
print(df[var2].value_counts(normalize=False), '\n')

# Cross Tabulation
print(pd.crosstab(df[var], df[var2]))