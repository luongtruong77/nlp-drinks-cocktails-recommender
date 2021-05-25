import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF, TruncatedSVD, LatentDirichletAllocation

import glob
from IPython.display import Image
from IPython.core.display import HTML
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from time import time
import random

with open('pkls/full_df.pkl', 'rb') as rf:
    full_df = pickle.load(rf)
with open('pkls/topics_by_description.pkl', 'rb') as rf:
    topics_by_description_df = pickle.load(rf)
with open('pkls/topics_by_description_matrix.pkl', 'rb') as rf:
    topics_by_description_matrix = pickle.load(rf)
with open('pkls/tfidf_description.pkl', 'rb') as rf:
    tfidf_des = pickle.load(rf)
with open('pkls/nmf_description.pkl', 'rb') as rf:
    nmf_des = pickle.load(rf)
with open('pkls/topics_by_tasting.pkl', 'rb') as rf:
    topics_by_tasting_df = pickle.load(rf)
with open('pkls/topics_by_tasting_matrix.pkl', 'rb') as rf:
    topics_by_tasting_matrix = pickle.load(rf)
with open('pkls/tfidf_tasting.pkl', 'rb') as rf:
    tfidf_tas = pickle.load(rf)
with open('pkls/nmf_tasting.pkl', 'rb') as rf:
    nmf_tas = pickle.load(rf)


topics_by_tasting_df = full_df.merge(topics_by_tasting_df, on='Name')

topics_by_description_df = full_df.merge(topics_by_description_df, on='Description')

########################################

with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center; color: black;'>Your personal drinks recommendation system!</h1>", unsafe_allow_html=True)
st.write('---')

########################################

options = st.selectbox('Please choose one of the options on how to search for your drinks.',
                       ('Please drink responsibly', 'Recommend by Description', 'Recommend By Tasting Info', 'Cocktails Recommendation'))


if options == 'Please drink responsibly':
    st.write("Please don't drink and drive!")

elif options == 'Recommend by Description':

    st.write('What would you like to drink today?')
    user_input = st.text_input("Anything that you would like to describe your drink (cheap scotch, sweet wine, ipa, etc.)")

    if user_input == '':
        st.write('Please drink responsibly!')
    else:

        topic_prob_dist = nmf_des.transform(tfidf_des.transform([user_input]))

        list_top_items_by_indices = list(cosine_similarity(topic_prob_dist, topics_by_description_matrix).argsort())[0][-1:-21:-1]

        top_10_random_items = topics_by_description_df.iloc[list_top_items_by_indices].sample(10)

        top_items = top_10_random_items[:3]

        next_items = top_10_random_items[3:8]

        st.write('**The top results:**')
        st.write('\n')
        for i in range(len(top_items)):
            st.write('**Name:** {}\n\n**Country:** {}\n\n**Alcohol Volume:** {}\n\n**Aroma:** {}\n\n**Flavor:** {}\n\n'
                     '**Price:** ${}\n\n**Comments:** {}\n\n*{}*'.
                     format(top_items.iloc[i].Name, top_items.iloc[i].Country, top_items.iloc[i].Alcohol_Vol,
                            top_items.iloc[i].Aroma, top_items.iloc[i].Flavor, top_items.iloc[i].Price,
                            top_items.iloc[i].Bottom_Line, top_items.iloc[i].Review))
            try:
                st.image([top_items.iloc[i].Photo_Link_2, top_items.iloc[i].Photo_Link], width=150)
            except:
                st.write('Image is not available!')
            st.write('---')

        my_expander = st.beta_expander('Show more recommendations')


        with my_expander:
            for i in range(len(next_items)):
                st.write('**Name:** {}\n\n**Country:** {}\n\n**Alcohol Volume:** {}\n\n**Aroma:** {}\n\n**Flavor:** {}\n\n'
                         '**Price:** ${}\n\n**Comments:** {}\n\n*{}*'.
                         format(next_items.iloc[i].Name, next_items.iloc[i].Country, next_items.iloc[i].Alcohol_Vol,
                                next_items.iloc[i].Aroma, next_items.iloc[i].Flavor, next_items.iloc[i].Price,
                                next_items.iloc[i].Bottom_Line, next_items.iloc[i].Review))
                try:
                    st.image([next_items.iloc[i].Photo_Link_2, next_items.iloc[i].Photo_Link], width=150)
                except:
                    st.write('Image is not available!')

                st.write('---')


elif options == 'Recommend By Tasting Info':

    st.write('What kind of food are your pairing with?')
    user_input = st.text_input("Anything that you would like to describe your food (steak, chicken, grilled, etc.)")

    if user_input == '':
        st.write('Please drink responsibly!')

    else:

        topic_prob_dist = nmf_tas.transform(tfidf_tas.transform([user_input]))

        list_top_items_by_indices = list(cosine_similarity(topic_prob_dist, topics_by_tasting_matrix).argsort())[0][
                                    -1:-21:-1]

        top_10_random_items = topics_by_tasting_df.iloc[list_top_items_by_indices].sample(10)

        top_items = top_10_random_items[:3]

        next_items = top_10_random_items[3:8]

        st.write('**The top results:**')
        st.write('\n')

        for i in range(len(top_items)):
            st.write('**Name:** {}\n\n**Country:** {}\n\n**Alcohol Volume:** {}\n\n**Style:** {}\n\n**Flavor:** {}\n\n'
                         '**Price:** ${}\n\n**Enjoy:** {}\n\n**Pairing:** {}\n\n*{}*'.
                     format(top_items.iloc[i].Name, top_items.iloc[i].Country, top_items.iloc[i].Alcohol_Vol,
                            top_items.iloc[i].Style, top_items.iloc[i].Flavor, top_items.iloc[i].Price,
                            top_items.iloc[i].Enjoy, top_items.iloc[i].Pairing, top_items.iloc[i].Review))
            try:
                st.image([top_items.iloc[i].Photo_Link_2, top_items.iloc[i].Photo_Link], width=150)
            except:
                st.write('Image is not available!')
            st.write('---')

        my_expander = st.beta_expander('Show more recommendations')


        with my_expander:
            for i in range(len(next_items)):
                st.write('**Name:** {}\n\n**Country:** {}\n\n**Alcohol Volume:** {}\n\n**Style:** {}\n\n**Flavor:** {}\n\n'
                         '**Price:** ${}\n\n**Enjoy:** {}\n\n**Pairing:** {}\n\n*{}*'.
                         format(next_items.iloc[i].Name, next_items.iloc[i].Country, next_items.iloc[i].Alcohol_Vol,
                                next_items.iloc[i].Style, next_items.iloc[i].Flavor, next_items.iloc[i].Price,
                                next_items.iloc[i].Enjoy, next_items.iloc[i].Pairing, next_items.iloc[i].Review))
                try:
                    st.image([next_items.iloc[i].Photo_Link_2, next_items.iloc[i].Photo_Link], width=150)
                except:
                    st.write('Image is not available!')

                st.write('---')

elif options == 'Cocktails Recommendation':
    st.write('Coming Soon...')





















