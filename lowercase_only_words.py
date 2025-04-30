#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 13:18:13 2025

@author: alex
"""
import pandas as pd
import re


city = 'puebla'
df = pd.read_csv(f'./tripAdvisor/{city}/{city}_all.csv')


df['Review'] = df['Review'].str.replace('_', ' ')
df['Review'] = df['Review'].str.replace(',', '')
df['Review'] = df['Review'].str.lower()
#df['Review'].str.replace(r'[^\w\s]', '')
df['Review'] = df['Review'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
df['Review'] = df['Review'].apply(lambda x: re.sub(r'\d+', '', x))
df['Review'] = df['Review'].apply(lambda x: re.sub(r'[´`]', '', x))
#df['Review'].str.replace('\´','')

lower_df = pd.DataFrame({'review': df['Review']})
lower_df.to_csv(f'./tripAdvisor/{city}/{city}_all_for_bertopic.csv', index=False)