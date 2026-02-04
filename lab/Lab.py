"""
Lab 1: Web Scraping and Data Analysis
In this lab I am scraping data from craigslist for cell phone listings on the east coast from Washington D.C to New York City.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import requests # Page requests
from bs4 import BeautifulSoup # HTML parsing

header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0'} # How we wish to appear to CL
url = 'https://reading.craigslist.org/search/morgantown-pa/moa?lat=40.1563&lon=-75.8554&search_distance=198#search=2~gallery~0' # The page we want to scrape
raw = requests.get(url,headers=header) # Get page


bsObj = BeautifulSoup(raw.text, 'html.parser') # Parse HTML
listings = bsObj.find_all(class_='cl-static-search-result') # Find all listings

import re # Regular expressions

brands = ['iphone', 'samsung', 'google', 'oneplus', 'nokia', 'sony', 'lg', 'htc', 'motorola', 'huawei', 'xiaomi']
models = [
    # Apple
    'iPhone 3G', 'iPhone 3GS', 'iPhone 4', 'iPhone 4S',
    'iPhone 5', 'iPhone 5C', 'iPhone 5S', 'iPhone 6', 'iPhone 6 Plus',
    'iPhone 6S', 'iPhone 6S Plus', 'iPhone SE (1st Gen)',
    'iPhone 7', 'iPhone 7 Plus', 'iPhone 8', 'iPhone 8 Plus',
    'iPhone X', 'iPhone XR', 'iPhone XS', 'iPhone XS Max',
    'iPhone 11', 'iPhone 11 Pro', 'iPhone 11 Pro Max',
    'iPhone SE (2nd Gen)',
    'iPhone 12', 'iPhone 12 Mini', 'iPhone 12 Pro', 'iPhone 12 Pro Max',
    'iPhone 13', 'iPhone 13 Mini', 'iPhone 13 Pro', 'iPhone 13 Pro Max',
    'iPhone SE (3rd Gen)',
    'iPhone 14', 'iPhone 14 Plus', 'iPhone 14 Pro', 'iPhone 14 Pro Max',
    'iPhone 15', 'iPhone 15 Plus', 'iPhone 15 Pro', 'iPhone 15 Pro Max',
    'iPhone 16', 'iPhone 16 Plus', 'iPhone 16 Pro', 'iPhone 16 Pro Max',
    'iPhone 17', 'iPhone 17 Plus', 'iPhone 17 Pro', 'iPhone 17 Pro Max',

    # Samsung
    'Galaxy S', 'Galaxy S2', 'Galaxy S3', 'Galaxy S4', 'Galaxy S5',
    'Galaxy S6', 'Galaxy S6 Edge', 'Galaxy S7', 'Galaxy S7 Edge',
    'Galaxy S8', 'Galaxy S8 Plus', 'Galaxy S9', 'Galaxy S9 Plus',
    'Galaxy S10', 'Galaxy S10e', 'Galaxy S10 Plus', 'Galaxy S20',
    'Galaxy S20 Plus', 'Galaxy S20 Ultra', 'Galaxy S21',
    'Galaxy S21 Plus', 'Galaxy S21 Ultra', 'Galaxy S22',
    'Galaxy S22 Plus', 'Galaxy S22 Ultra', 'Galaxy S23',
    'Galaxy S23 Plus', 'Galaxy S23 Ultra', 'Galaxy S24',
    'Galaxy S24 Plus', 'Galaxy S24 Ultra',
    'Galaxy S25', 'Galaxy S25 Plus', 'Galaxy S25 Ultra',
    'Galaxy Note 3', 'Galaxy Note 4', 'Galaxy Note 5',
    'Galaxy Note 8', 'Galaxy Note 9', 'Galaxy Note 10',
    'Galaxy Note 10 Plus',
    'Galaxy Z Flip', 'Galaxy Z Flip 3', 'Galaxy Z Flip 4',
    'Galaxy Z Flip 5', 'Galaxy Z Flip 6',
    'Galaxy Z Fold', 'Galaxy Z Fold 2', 'Galaxy Z Fold 3',
    'Galaxy Z Fold 4', 'Galaxy Z Fold 5', 'Galaxy Z Fold 6',
    

    # Google
    'Pixel XL', 'Pixel 2', 'Pixel 2 XL', 'Pixel 3',
    'Pixel 3 XL', 'Pixel 4', 'Pixel 4 XL', 'Pixel 5',
    'Pixel 6', 'Pixel 6 Pro', 'Pixel 7', 'Pixel 7 Pro',
    'Pixel 8', 'Pixel 8 Pro',
    'Pixel 9', 'Pixel 9 Pro', 'Pixel 9 Pro XL',

    # OnePlus
    'OnePlus One', 'OnePlus 2', 'OnePlus 3', 'OnePlus 3T',
    'OnePlus 5', 'OnePlus 5T', 'OnePlus 6', 'OnePlus 6T',
    'OnePlus 7', 'OnePlus 7 Pro', 'OnePlus 7T',
    'OnePlus 8', 'OnePlus 8 Pro', 'OnePlus 9', 'OnePlus 9 Pro',
    'OnePlus 10 Pro', 'OnePlus 11', 'OnePlus 12', 'OnePlus 13',

    # Nokia
    'Nokia 3310', 'Nokia 1100', 'Nokia N95', 'Nokia E71',
    'Lumia 520', 'Lumia 620', 'Lumia 920', 'Lumia 1020',
    'Nokia 6', 'Nokia 7 Plus', 'Nokia 8', 'Nokia 9 PureView',
    'Nokia X30', 'Nokia G60',

    # Sony
    'Xperia Z', 'Xperia Z1', 'Xperia Z2', 'Xperia Z3',
    'Xperia Z5', 'Xperia X', 'Xperia XZ', 'Xperia XZ1',
    'Xperia XZ2', 'Xperia 1', 'Xperia 1 II', 'Xperia 1 III',
    'Xperia 1 IV', 'Xperia 1 V',
    'Xperia 5', 'Xperia 5 II', 'Xperia 5 III', 'Xperia 10',

    # LG
    'LG G2', 'LG G3', 'LG G4', 'LG G5', 'LG G6', 'LG G7',
    'LG V10', 'LG V20', 'LG V30', 'LG V40', 'LG V50',
    'LG Velvet', 'LG Wing',

    # HTC
    'HTC One', 'HTC One M7', 'HTC One M8', 'HTC One M9',
    'HTC 10', 'HTC U11', 'HTC U12 Plus', 'HTC Desire',

    # Motorola
    'Moto G', 'Moto G2', 'Moto G3', 'Moto G4', 'Moto G5',
    'Moto G6', 'Moto G7', 'Moto G Power',
    'Moto X', 'Moto X Style', 'Moto X Play',
    'Moto Z', 'Moto Z Play',
    'Motorola Razr', 'Motorola Razr V3', 'Motorola Razr 2023',

    # Huawei
    'Huawei P8', 'Huawei P9', 'Huawei P10', 'Huawei P20',
    'Huawei P30', 'Huawei P30 Pro', 'Huawei P40', 'Huawei P40 Pro',
    'Huawei Mate 10', 'Huawei Mate 20', 'Huawei Mate 30',
    'Huawei Mate 40', 'Huawei Mate 50', 'Huawei Mate 60',

    # Xiaomi
    'Xiaomi Mi 5', 'Xiaomi Mi 6', 'Xiaomi Mi 8', 'Xiaomi Mi 9',
    'Xiaomi Mi 10', 'Xiaomi Mi 11', 'Xiaomi Mi 12', 'Xiaomi 13',
    'Xiaomi 14', 'Xiaomi 14 Pro',
    'Redmi Note 7', 'Redmi Note 8', 'Redmi Note 9',
    'Redmi Note 10', 'Redmi Note 11', 'Redmi Note 12', 'Redmi Note 13'
]

data = []
for k in range( len(listings) ):
    # Extract title, price, link, and location:
    title = listings[k].find('div',class_='title').get_text().lower()
    price = listings[k].find('div',class_='price').get_text()
    link = listings[k].find(href=True)['href']
    location = listings[k].find('div',class_='location').get_text().lower() if listings[k].find('div',class_='location') else np.nan

    # Get brand from the title string:
    words = title.split()
    hits = [word for word in words if word in brands] # Find brands in the title
    if len(hits) == 0:
        brand = 'missing'
    else:
        brand = hits[0]

    # Get model from the title string, looking for longest matches first
    hits = sorted(
    (model for model in models if model.lower() in title.lower()),
    key=len,
    reverse=True
    )
    if len(hits) == 0:
        model = 'missing'
    else:
        model = hits[0]     

    # Adding to data list:
    data.append({'title':title, 'price':price, 'link':link, 'brand':brand, 'model':model, 'location':location})

# Wrangle the Data
df = pd.DataFrame.from_dict(data)
df['price'] = df['price'].str.replace('$','')
df['price'] = df['price'].str.replace(',','')
df['price'] = pd.to_numeric(df['price'],errors='coerce')
df.loc[df['price'] <= 10, 'price'] = np.nan # Remove listings with price <= $10, likely not cell phones or errors
print(df.shape, '\n')
df.to_csv('craigslist_cell_phones.csv')
df.head()

# Showing value counts for brand and models
print(df['brand'].value_counts(), '\n')
print(df['model'].value_counts(), '\n')

# EDA for price
print(df['price'].describe())
df['price'].hist(grid=False)
plt.xlabel('Price ($)')
plt.ylabel('Number of Listings')
plt.title('Histogram of Cell Phone Prices on Craigslist')
plt.show()

# Price by brand:
df.loc[:,['price','brand']].groupby('brand').describe()
plt.figure(figsize=(12,6))
sns.boxplot(x='brand', y='price', data=df)
plt.ylim(0,1200)
plt.xlabel('Brand')
plt.ylabel('Price ($)')
plt.title('Boxplot of Cell Phone Prices by Brand on Craigslist')
plt.show()

# Price by model (top 10 models):
top_models = df['model'].value_counts().nlargest(10).index
plt.figure(figsize=(12,6))
sns.boxplot(x='model', y='price', data=df[df['model'].isin(top_models)])
plt.ylim(0,800)
plt.xlabel('Model')
plt.ylabel('Price ($)')
plt.title('Boxplot of Cell Phone Prices by Top 10 Models on Craigslist')
plt.show()

# Price by location (top 5 locations):
top_locations = df['location'].value_counts().nlargest(5).index
plt.figure(figsize=(12,6))
sns.boxplot(x='location', y='price', data=df[df['location'].isin(top_locations)])
plt.ylim(0,1000)
plt.xlabel('Location')
plt.ylabel('Price ($)')
plt.title('Boxplot of Cell Phone Prices by Top 5 Locations on Craigslist')
plt.show()