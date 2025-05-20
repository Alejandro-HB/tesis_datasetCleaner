#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 13:18:13 2025

@author: alex
"""
import pandas as pd
import re


city = 'puebla'
# For image description csv
model = 'places'
#df = pd.read_csv(f'./tripAdvisor/{city}/{city}_all.csv')
df = pd.read_csv(f'./data/lemmatized_data/{city}_lemmatized_{model}.csv')


# Column name lemmatized for image description csv, Review for tripAdvisor reviews
col = 'lemmatized'

df[col] = df[col].str.replace('_', '')
df[col] = df[col].str.replace(',', '')
df[col] = df[col].str.lower()
#df['Review'].str.replace(r'[^\w\s]', '')
df[col] = df[col].apply(lambda x: re.sub(r'[^\w\s]', '', x))
df[col] = df[col].apply(lambda x: re.sub(r'\d+', '', x))
df[col] = df[col].apply(lambda x: re.sub(r'[´`]', '', x))
#df['Review'].str.replace('\´','')

lower_df = pd.DataFrame({col: df[col]})

#lower_df.to_csv(f'./tripAdvisor/{city}/{city}_all_for_bertopic.csv', index=False)
lower_df.to_csv(f'./data/lemmatized_data/lemmatized_no_numbers/{city}_lemmatized_{model}.csv', index=False)