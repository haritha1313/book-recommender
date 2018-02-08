# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 19:15:29 2018

@author: pegasus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import warnings
from sklearn.decomposition import TruncatedSVD
from itertools import starmap

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


usc_rating_pivot = us_canada_user_rating.pivot(index = 'userID', columns = 'bookTitle', values = 'bookRating').fillna(0)
X = usc_rating_pivot.values.T

SVD = TruncatedSVD(n_components = 12, random_state = 17)
matrix = SVD.fit_transform(X)
corr = np.corrcoef(matrix)

us_canada_book_title = usc_rating_pivot.columns
us_canada_book_list = list(us_canada_book_title)
bindex = us_canada_book_list.index(args['name'])

corr_bindex  = corr[bindex]
arr = [str(b) for b in list(us_canada_book_title[(corr_bindex<1.0) & (corr_bindex>0.95)])]
arr = arr[0:9]
print('Recommendations for {}:'.format(args['name']))
print('\n'.join(starmap('{}: {}'.format, enumerate(arr))))