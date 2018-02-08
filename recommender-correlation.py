# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 12:12:41 2018

@author: pegasus
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import warnings

warnings.simplefilter(action='ignore')
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required = True, help = "Name of book you enjoyed")
args = vars(ap.parse_args())

books = pd.read_csv('../data/BX-Books.csv', sep=';', error_bad_lines = False, encoding = 'latin-1', warn_bad_lines=False)
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
users = pd.read_csv('../data/BX-Users.csv', sep=';', error_bad_lines = False, encoding = 'latin-1', warn_bad_lines=False)
users.columns = ['userID', 'Location', 'Age']
ratings = pd.read_csv('../data/BX-Book-Ratings.csv', sep = ';', error_bad_lines = False, encoding = 'latin-1', warn_bad_lines=False)
ratings.columns = ['userID', 'ISBN', 'bookRating']

isbn = books[books['bookTitle'] == args['name']]['ISBN'].to_string(index=False)
print("Wondering what you would enjoy...")
"""
#Analyzing ratings
print(ratings.shape)
print(list(ratings.columns))
print(ratings.head())

#Book rating graph
plt.rc("font", size=15)
ratings.bookRating.value_counts(sort=False).plot(kind='bar')
plt.title('Rating-analysis')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('rating-analysis.png', bbox_inches = 'tight')
plt.show()
"""

rating_count = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].count())
#print(rating_count.sort_values('bookRating', ascending=False).head())

average_rating = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].mean())
average_rating['ratingCount'] = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].count())
#print(average_rating.sort_values('ratingCount', ascending=False).head())

counts1 = ratings['userID'].value_counts()
ratings = ratings[ratings['userID'].isin(counts1[counts1>=200].index)]
counts = ratings['bookRating'].value_counts()
ratings = ratings[ratings['bookRating'].isin(counts[counts>=100].index)]

ratings_pivot = ratings.pivot(index = 'userID', columns = 'ISBN').bookRating
userID = ratings_pivot.index
ISBN = ratings_pivot.columns
#print(ratings_pivot.shape)

bones_ratings = ratings_pivot[isbn]
similar_to_bones = ratings_pivot.corrwith(bones_ratings)
corr_bones = pd.DataFrame(similar_to_bones, columns = ['pearsonR'])
corr_bones.dropna(inplace=True)
corr_summary = corr_bones.join(average_rating['ratingCount'])
top10 = list((corr_summary[corr_summary['ratingCount']>=300].sort_values('pearsonR', ascending=False).head(10)).index)

top10 = [str(r) for r in top10]

books_corr_to_bones = pd.DataFrame(top10, index=np.arange(10), columns=['ISBN'])
corr_books = pd.merge(books_corr_to_bones, books, on='ISBN')
print("Found 'em: \n")
print(corr_books.filter(['bookTitle','bookAuthor']))
