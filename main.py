import pandas as pd
import seaborn as sns
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

#%matplotlib.inline
matplotlib.rcParams['figure.figsize'] = (12, 8) # adjust configuration of plots

df = pd.read_csv(r'C:\Users\77471\Desktop\pet-projects\movie-industry\movies.csv')
print(df.head())

#Checking for percentage of NULL values
'''
for col in df.columns:
    prcnt_missing  = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, prcnt_missing))
'''

df = df.fillna(0)

#Data Cleaning
#print(df.dtypes)
df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')
#print(df.dtypes)

#Created column with correct year of released from release date
df['correct_year'] = df['released'].astype('str').str[:4]
print(df.head())

df.sort_values(by=['gross'], inplace=False, ascending=False)
print(df.head())

#pd.set_option('display.max_rows', None) #Display all the data

#Drop duplicates
df['company'].drop_duplicates() #sort_values(ascending=False)

# Correlation between budget and gross
# Scatter plot

plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Budget VS Gross')
plt.xlabel('Budget')
plt.ylabel('Gross')
sns.regplot(x='budget', y='gross', data = df, scatter_kws = {"color":"pink"}, line_kws = {"color":"cyan"})
plt.show()

# Show correlation between numerical fields: Pearson (default correlation), Kendall, Spearman
correlation_matrix = df.corr(method = 'pearson', numeric_only=True)
sns.heatmap(correlation_matrix, annot = True)
plt.title('Correlation matrix between og numeric movie attibutes')
plt.xlabel('Movie Attibutes')
plt.ylabel('Movie Attibutes')
plt.show()

# Correlation between company and gross
df_num = df
for col_name in df_num.columns:
    if(df_num[col_name].dtype == 'object'):
        df_num[col_name] = df_num[col_name].astype('category')
        df_num[col_name] = df_num[col_name].cat.codes

print(df_num)

correlation_matrix = df_num.corr(method = 'pearson', numeric_only=True)
sns.heatmap(correlation_matrix, annot = True)
plt.title('Correlation matrix between all movie attibutes')
plt.xlabel('Movie Attibutes')
plt.ylabel('Movie Attibutes')
plt.show()

print(df.corr(method = 'pearson', numeric_only=True))

#Unstacking
correlation_matrix = df_num.corr(method = 'pearson', numeric_only=True)
corr_pairs = correlation_matrix.unstack()
sorted_pairs = corr_pairs.sort_values()
print(sorted_pairs)
high_corr = sorted_pairs[(sorted_pairs) > 0.5]
print(high_corr)