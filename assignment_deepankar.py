#! /usr/bin/env python

# Author: Deepankar Kotnala 
# Batch: DS_C7_UpGrad

# Importing the libraries
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


warnings.filterwarnings('ignore')
np.random.seed(8)

def calculate_loan(df_qty, df_cost):
    df_total_cost = df_qty*df_cost
    return round(sum(df_total_cost),2)


if __name__ == '__main__':

    customer_features = pd.read_csv(sys.argv[1])
    last_month_assortment =pd.read_csv(sys.argv[2])
    next_month_assortment = pd.read_csv(sys.argv[3])
    next_purchase_order = pd.read_csv(sys.argv[4])
    original_purchase_order = pd.read_csv(sys.argv[5])
    product_features = pd.read_csv(sys.argv[6])
    
    customer_features['favorite_genres'] = customer_features.favorite_genres.apply(lambda x: x.lower())
    customer_features['favorite_genres'] = customer_features.favorite_genres.apply(lambda x: x.replace("-", ""))
    
    cnv = CountVectorizer()
    vect = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
    X_vect = vect.fit_transform(customer_features.favorite_genres)
    genres_data = pd.DataFrame(X_vect.toarray(), columns=vect.get_feature_names())
    customer_features = customer_features.join(genres_data)
    to_drop = ['favorite_genres']
    customer_features.drop(to_drop, inplace=True, axis=1)
    
    last_month_assortment['shipping_cost'] = last_month_assortment.purchased.apply(lambda x: 0.6 if x==True else 1.2)
    master_df = pd.merge(product_features, last_month_assortment, on ='product_id')
    master_df = pd.merge(master_df, original_purchase_order, on ='product_id')
    master_df = pd.merge(master_df, customer_features, on='customer_id')
    
    prev_month_cost = round(sum(master_df['shipping_cost']),2)
    prev_month_loan = calculate_loan(original_purchase_order['quantity_purchased'], original_purchase_order['cost_to_buy'])
    next_month_cost = calculate_loan(next_purchase_order['quantity_purchased'], next_purchase_order['cost_to_buy'])
 
    target='purchased'
    le = LabelEncoder()
    master_df['age_bucket'] = master_df['age_bucket'].astype(str)
    master_df['age_bucket'] = le.fit_transform(master_df['age_bucket'])
    master_df['fiction'] = le.fit_transform(master_df['fiction'])
    master_df['genre'] = le.fit_transform(master_df['genre'])
    master_df['is_returning_customer'] = le.fit_transform(master_df['is_returning_customer'])
    master_df['purchased'] = le.fit_transform(master_df['purchased'])
    
    predictors = ['retail_value', 'length', 'difficulty','fiction', 'genre', 'age_bucket', 
                'is_returning_customer', 'beachread', 'biography', 'classic', 'drama', 'history', 
                'poppsychology', 'popsci', 'romance', 'scifi', 'selfhelp', 'thriller']

    X = master_df[predictors]
    y = master_df.loc[:, master_df.columns == target] 
    
    xgb_model = XGBClassifier(C=1)
    xgb_model.fit(X, y)
    
    temp_df = pd.DataFrame(last_month_assortment.groupby(['product_id'])['purchased'].sum().reset_index())
    prev_month_books_remaining = pd.merge(original_purchase_order, temp_df, on = 'product_id')
    prev_month_books_remaining['quantity_remaining'] = prev_month_books_remaining['quantity_purchased'] - prev_month_books_remaining['purchased']
    prev_month_books_remaining = prev_month_books_remaining.drop(columns = ['quantity_purchased', 'purchased'])
    total_sales_prev_month= round(sum(master_df['retail_value'].where(master_df['purchased']==True, 0)),2)
    next_month_pred = pd.merge(next_month_assortment, prev_month_books_remaining, on = 'product_id')
    next_month_pred = pd.merge(next_month_pred, customer_features, on = 'customer_id')
    next_month_pred = pd.merge(next_month_pred, product_features, on = 'product_id')
    
    next_month_pred['age_bucket'] = next_month_pred['age_bucket'].astype(str)
    next_month_pred['fiction'] = le.fit_transform(next_month_pred['fiction'])
    next_month_pred['genre'] = le.fit_transform(next_month_pred['genre'])
    next_month_pred['age_bucket'] = le.fit_transform(next_month_pred['age_bucket'])
    next_month_pred['is_returning_customer'] = le.fit_transform(next_month_pred['is_returning_customer'])
    
    features_for_prediction = ['retail_value', 'length', 'difficulty','fiction', 'genre', 'age_bucket', 
                'is_returning_customer', 'beachread', 'biography', 'classic', 'drama', 'history', 
                'poppsychology', 'popsci', 'romance', 'scifi', 'selfhelp', 'thriller']
    X = next_month_pred[features_for_prediction]
    
    predict_next_month_purchase = xgb_model.predict(X)
    next_month_shipping = (sum(predict_next_month_purchase)*0.6 + (X.shape[0]-sum(predict_next_month_purchase)*1.2))
    next_month_pred['next_month_purchase_predictions'] = predict_next_month_purchase
    next_sales = round(sum(next_month_pred['retail_value'].where(next_month_pred['next_month_purchase_predictions']==1, 0)),2)
    tot_sales = total_sales_prev_month + next_sales
    tot_cost = prev_month_loan + next_month_cost + prev_month_cost + next_month_shipping
    Final_Decision = ('Yes' if (tot_sales - tot_cost > 0) else 'No')
    print(Final_Decision)

    
