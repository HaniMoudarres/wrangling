""" Question 2 """

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

""" Step 1 """

# Load the data
df = pd.read_excel('/Users/hanimoudarres/Downloads/Foundations of ML/HW 2/wrangling/assignment/data/shark_data.xls')

""" Step 2 """

# Initial columns
print(df.columns, len(df.columns))
print('\n')

# Drop columns that are completely empty
df = df.dropna(axis=1, how='all')

# Columns after dropping empty ones
# (it stayed the same in this case)
print(df.columns, len(df.columns))
print('\n')

""" Step 3 """

var = 'Year'

# Coerce to numeric, forcing errors to NaN
df[var] = pd.to_numeric(df[var], errors='coerce')

# Look at range
print(df[var].describe(), '\n')

# Filter for attacks since 1940
df_recent = df[df[var] >= 1940]

# Plot number of attacks per year since 1940
attacks_per_year = df_recent[var].value_counts().sort_index()
attacks_per_year.plot(figsize=(12,5), title="Shark attacks since 1940")
plt.xlabel("Year")
plt.ylabel("Number of attacks")
plt.show()

""" Step 4 """

# Remove non-numeric characters and convert to float
df_recent['Age'] = pd.to_numeric(df_recent['Age'], errors='coerce')

# Plot histogram
plt.figure(figsize=(10,5))
df_recent['Age'].hist(bins=30)
plt.title("Histogram of Victim Ages")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

""" Step 5 """

# Clean Gender variable
df_recent['Sex'] = df_recent['Sex'].str.strip().str.upper()  # Remove whitespace, uppercase

# Shows M, F, M X 2, and LLI --> need to fix
print(df_recent['Sex'].value_counts(dropna=False), '\n')

sex_mapping = {
    'M': 'M',
    'F': 'F',
    'M X 2': 'M',     # assume these are males
    'LLI': 'Unknown'  # treat as unknown/missing
}
df_recent['Sex'] = df_recent['Sex'].str.strip().map(sex_mapping).fillna('Unknown')

# Check distribution after cleaning
print(df_recent['Sex'].value_counts(dropna=False), '\n')

# Compute proportion male
male_prop = (df_recent['Sex'] == 'M').mean()
print(f"Proportion of male victims: {male_prop:.2f}\n")


""" Step 6 """

# Standardize Type variable
df_recent['Type'] = df_recent['Type'].str.strip().str.capitalize()

# Map all types to Provoked, Unprovoked, Unknown
type_mapping = {
    'Unprovoked': 'Unprovoked',
    'Provoked': 'Provoked',
}
df_recent['Type'] = df_recent['Type'].map(type_mapping).fillna('Unknown')

# Proportion unprovoked
unprovoked_prop = (df_recent['Type'] == 'Unprovoked').mean()
print(f"Proportion of unprovoked attacks: {unprovoked_prop:.2f}\n")


""" Step 7 """

# Standardize Fatal variable
df_recent['Fatal Y/N'] = df_recent['Fatal Y/N'].str.strip().str.upper()

# Map to Y, N, Unknown
fatal_mapping = {
    'Y': 'Y', 
    'N': 'N'
}

df_recent['Fatal Y/N'] = df_recent['Fatal Y/N'].map(fatal_mapping).fillna('Unknown')

# Check distribution
print(df_recent['Fatal Y/N'].value_counts(normalize=False), '\n')

""" Step 8 """

# Are unprovoked attacks more likely on men or women?
print(pd.crosstab(df_recent['Sex'], df_recent['Type'], normalize='index'))
print('\n')

# Fatality by attack type
print(pd.crosstab(df_recent['Type'], df_recent['Fatal Y/N'], normalize='index'))
print('\n')

# Fatality by sex
print(pd.crosstab(df_recent['Sex'], df_recent['Fatal Y/N'], normalize='index'))
print('\n')

""" Step 9 """

# Make lowercase and split text into words
df_recent['Species '] = df_recent['Species '].str.lower().fillna('')
species_lists = df_recent['Species '].str.split()

# Check for white shark
white_shark_prop = species_lists.apply(lambda x: 'white' in x and 'shark' in x).mean()
print(f"Proportion of attacks by white sharks: {white_shark_prop:.2f}")