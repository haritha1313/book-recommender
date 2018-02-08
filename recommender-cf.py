# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 18:14:38 2018

@author: pegasus
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import warnings
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

warnings.simplefilter(action='ignore')

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required = True, help = "Name of book you enjoyed")
args = vars(ap.parse_args())

book = pd.read_csv('../data/BX-Books.csv', sep=';', error_bad_lines = False, encoding = 'latin-1', warn_bad_lines=False)
book.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
user = pd.read_csv('../data/BX-Users.csv', sep=';', error_bad_lines = False, encoding = 'latin-1', warn_bad_lines=False)
user.columns = ['userID', 'Location', 'Age']
rating = pd.read_csv('../data/BX-Book-Ratings.csv', sep = ';', error_bad_lines = False, encoding = 'latin-1', warn_bad_lines=False)
rating.columns = ['userID', 'ISBN', 'bookRating']

#indexno = book[book['bookTitle'] == args['name']]['ISBN'].index
print("\nWondering what you would enjoy...")

combine_book_rating = pd.merge(rating, book, on = 'ISBN')

columns = ['yearOfPublication', 'publisher', 'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
combine_book_rating = combine_book_rating.drop(columns, axis=1)

combine_book_rating = combine_book_rating.dropna(axis=0, subset = ['bookTitle'])
book_ratingCount = (combine_book_rating.
                    groupby(by = ['bookTitle'])
                    ['bookRating'].
                    count().
                    reset_index().
                    rename(columns = {'bookRating': 'totalRatingCount'})
                    [['bookTitle', 'totalRatingCount']]
                   )
                   
#print(book_ratingCount.head())

rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'bookTitle', right_on = 'bookTitle', how='left')
#print(rating_with_totalRatingCount.head())

pd.set_option('display.float_format', lambda x: '%.3f' %x)
#print(book_ratingCount['totalRatingCount'].describe())
#print(book_ratingCount['totalRatingCount'].quantile(np.arange(.9, 1, .01)))

###1% books have more than 50 rating, which is enough for now
popularity_threshold = 50
rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')

combined = rating_popular_book.merge(user, left_on = 'userID', right_on = 'userID', how = 'left')
us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada")]
us_canada_user_rating = us_canada_user_rating.drop('Age', axis=1)

us_canada_user_rating = us_canada_user_rating.drop_duplicates(['userID', 'bookTitle'])

usc_rating_pivot = us_canada_user_rating.pivot(index = 'bookTitle', columns = 'userID', values = 'bookRating').fillna(0)
usc_rating_matrix = csr_matrix(usc_rating_pivot.values)


model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(usc_rating_matrix)

distances, indices = model_knn.kneighbors(usc_rating_pivot.loc[args['name']].reshape(1, -1), n_neighbors = 11)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print("Recommendations for {}:\n".format(args['name']))
    else:
        print('{}: {}'.format(i, usc_rating_pivot.index[indices.flatten()[i]]))
