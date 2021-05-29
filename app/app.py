import streamlit as st
import pandas as pd
import pickle
from PIL import Image

pd.set_option('display.max_columns', None)
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics import edit_distance

with open('pkls/extended_df.pkl', 'rb') as rf:
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
with open('pkls/cocktails_with_photos.pkl', 'rb') as rf:
    cocktails_with_photos = pickle.load(rf)
with open('pkls/ingredients_df.pkl', 'rb') as rf:
    ingredients_df = pickle.load(rf)
with open('pkls/nmf_ingredients.pkl', 'rb') as rf:
    nmf_ingredients = pickle.load(rf)
with open('pkls/ingredients_matrix.pkl', 'rb') as rf:
    ingredients_matrix = pickle.load(rf)
with open('pkls/tfidf_ingredients.pkl', 'rb') as rf:
    tfidf_ingredients = pickle.load(rf)
full_df.fillna('N/A', inplace=True)
cocktails_with_photos.fillna('N/A', inplace=True)

topics_by_tasting_df = full_df.merge(topics_by_tasting_df, on='Name')

topics_by_description_df = full_df.merge(topics_by_description_df, on='Description')

google_search_query = 'https://www.google.com/search?q='

########################################
########################################


options = st.sidebar.selectbox('Please choose one of the options on how to search for your drinks.',
                               ('Home Page', 'Spirits, Wine, and Beer Recommendation',
                                'Cocktails Recommendation'))
st.sidebar.write('---')
st.sidebar.write(
    'This app was built by Steven Truong. Please reach out to me at [LinkedIn](https://www.linkedin.com/in/luongtruong77/).'
    'The source codes on how to build this app can be found here [Github](https://github.com/luongtruong77/nlp-drinks-cocktails-recommender)')

if options == 'Home Page':

    st.image(Image.open('figures/cocktails.jpg'))
    st.markdown(
        "<h1 style='text-align: center; color: black;'>Your personal drinks and cocktails recommendation system!</h1>",
        unsafe_allow_html=True)
    st.write('---')

    st.markdown(
        'This fun recommendation app was built using Natural Language Processing (NLP) with the data was acquired '
        'from multiple sources: [Tastings.com](https://www.tastings.com/Reviews/Latest-Spirits-Wine-Beer-Reviews.aspx), '
        '[CocktailDB](https://www.thecocktaildb.com/), [Liquor](https://www.liquor.com/cocktail-by-spirit-4779438), '
        'and [Caskers](https://www.caskers.com/spirits/).', unsafe_allow_html=True)
    st.write(
        'The reviews, descriptions, and tasting information (wine and spirits), as well as ingredients (cocktails) '
        'were encoded using *Term Frequency–Inverse Document Frequency (TF-IDF)* and used to build *Non-negative Matrix Factorization (NMF)*'
        ' topic modeling. Upon the topics extracted from the model, the *cosine similarity metric* is used to compare'
        ' users input information and return the most relevant products. Moreover, to recommend cocktails by name, '
        'the *levenshtein distance metric* is used to correct the input names.')
    st.write('Please choose how you want to be recommended on the left side bar.')
    st.write('---')
    st.write("**PLEASE DON'T DRINK AND DRIVE!**")
    st.write('---')
    st.image(Image.open('figures/pregnancy_warning.png'))
    st.write('\n')


elif options == 'Spirits, Wine, and Beer Recommendation':

    st.image(Image.open('figures/liquor.jpg'))

    choices = st.radio('Please choose how you want to be recommended by', ('By Description', 'By Tasting Info'))

    if choices == 'By Description':

        nmf = nmf_des
        tfidf = tfidf_des
        topics_matrix = topics_by_description_matrix
        topics_df = topics_by_description_df

        st.write('What would you like to drink today?')
        user_input = st.text_input(
            "Anything that you would like to describe your drink (scotch, spice rum, sweet wine, ipa, etc.)")

        if user_input == '':
            st.write('Please drink responsibly!')
            st.write('My personal favorites are "scotch" and "whiskey"')
        else:

            topic_prob_dist = nmf.transform(tfidf.transform([user_input]))
            list_top_items_by_indices = list(cosine_similarity(topic_prob_dist, topics_matrix).argsort())[0][
                                        -1:-31:-1]
            top_items = topics_df.iloc[list_top_items_by_indices].sample(10)

            st.write('**The top results:**')
            st.write('\n')
            for i in range(3):

                st.write(
                    '**Name:** {}\n\n**Country:** {}\n\n**Alcohol Volume:** {}\n\n**Aroma:** {}\n\n**Flavor:** {}\n\n'
                    '**Price:** ${}\n\n**Comments:** {}\n\nFind where to buy this product near you. Click [HERE!]({})\n\n*{}*'.
                        format(top_items.iloc[i].Name, top_items.iloc[i].Country, top_items.iloc[i].Alcohol_Vol,
                               top_items.iloc[i].Aroma, top_items.iloc[i].Flavor, top_items.iloc[i].Price,
                               top_items.iloc[i].Bottom_Line,
                               google_search_query + "+".join(top_items.iloc[i].Name.split(" ")),
                               top_items.iloc[i].Review))
                try:
                    st.image([top_items.iloc[i].Photo_Link_2, top_items.iloc[i].Photo_Link], width=200)
                except:
                    st.image(top_items.iloc[i].Photo_Link_2, width=200)

                st.write('---')

            my_expander = st.beta_expander('Show more recommendations')
            with my_expander:
                for i in range(3, 10):
                    st.write(
                        '**Name:** {}\n\n**Country:** {}\n\n**Alcohol Volume:** {}\n\n**Aroma:** {}\n\n**Flavor:** {}\n\n'
                        '**Price:** ${}\n\n**Comments:** {}\n\nFind where to buy this product near you. Click [HERE!]({})\n\n*{}*'.
                            format(top_items.iloc[i].Name, top_items.iloc[i].Country, top_items.iloc[i].Alcohol_Vol,
                                   top_items.iloc[i].Aroma, top_items.iloc[i].Flavor, top_items.iloc[i].Price,
                                   top_items.iloc[i].Bottom_Line,
                                   google_search_query + "+".join(top_items.iloc[i].Name.split(" ")),
                                   top_items.iloc[i].Review))
                    try:
                        st.image([top_items.iloc[i].Photo_Link_2, top_items.iloc[i].Photo_Link], width=200)
                    except:
                        st.image(top_items.iloc[i].Photo_Link_2, width=200)

                    st.write('---')

    elif choices == 'By Tasting Info':

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
                st.write(
                    '**Name:** {}\n\n**Country:** {}\n\n**Alcohol Volume:** {}\n\n**Style:** {}\n\n**Flavor:** {}\n\n'
                    '**Price:** ${}\n\n**Enjoy:** {}\n\n**Pairing:** {}\n\nFind where to buy this product near you. Click [HERE!]({})\n\n*{}*'.
                        format(top_items.iloc[i].Name, top_items.iloc[i].Country, top_items.iloc[i].Alcohol_Vol,
                               top_items.iloc[i].Style, top_items.iloc[i].Flavor, top_items.iloc[i].Price,
                               top_items.iloc[i].Enjoy, top_items.iloc[i].Pairing,
                               google_search_query + "+".join(top_items.iloc[i].Name.split(" ")),
                               top_items.iloc[i].Review))
                try:
                    st.image([top_items.iloc[i].Photo_Link_2, top_items.iloc[i].Photo_Link], width=150)
                except:
                    st.write('Image is not available!')
                st.write('---')

            my_expander = st.beta_expander('Show more recommendations')

            with my_expander:
                for i in range(len(next_items)):
                    st.write(
                        '**Name:** {}\n\n**Country:** {}\n\n**Alcohol Volume:** {}\n\n**Style:** {}\n\n**Flavor:** {}\n\n'
                        '**Price:** ${}\n\n**Enjoy:** {}\n\n**Pairing:** {}\n\nFind where to buy this product near you. Click [HERE!]({})\n\n*{}*'.
                            format(next_items.iloc[i].Name, next_items.iloc[i].Country, next_items.iloc[i].Alcohol_Vol,
                                   next_items.iloc[i].Style, next_items.iloc[i].Flavor, next_items.iloc[i].Price,
                                   next_items.iloc[i].Enjoy, next_items.iloc[i].Pairing,
                                   google_search_query + "+".join(next_items.iloc[i].Name.split(" ")),
                                   next_items.iloc[i].Review))
                    try:
                        st.image([next_items.iloc[i].Photo_Link_2, next_items.iloc[i].Photo_Link], width=150)
                    except:
                        st.write('Image is not available!')

                    st.write('---')

elif options == 'Cocktails Recommendation':

    st.image(Image.open('figures/cocktails2.jpg'))

    cocktails_options = st.radio('Please choose how you want to search:', ('By Name', 'By Ingredients'))

    if cocktails_options == 'By Name':

        user_input = st.text_input("case insensitive (gin and tonic, old-fashioned, etc.)")

        if user_input == '':
            st.write('My personal favorites are "Manhattan" and "old-fashioned"')

        else:

            n = len(user_input)

            distances = []

            for i in range(len(cocktails_with_photos)):
                dist = edit_distance(user_input.lower(), cocktails_with_photos.Name.iloc[i].strip().lower()[:n])

                distances.append(dist)

            cocktails_with_photos['Distance'] = pd.DataFrame(distances, columns={'Distance'}).Distance

            top_items = cocktails_with_photos.sort_values(by='Distance').head(10).fillna('N/A')

            st.write('**The top results:**')
            st.write('\n')

            for i in range(3):

                st.write(
                    '**Name:** {}\n\n**Category:** {}\n\n**Ingredients:** {}\n\n**Instructions:** {}\n\n**Serve in:** {}'
                        .format(top_items.iloc[i].Name, top_items.iloc[i].Category, top_items.iloc[i].Ingredients,
                                top_items.iloc[i].Instructions,
                                top_items.iloc[i].Serve_In))
                try:
                    st.image(top_items.iloc[i].Photo_Link, width=250)
                except:
                    st.write('Image is not available!')

                st.write('---')

            my_expander = st.beta_expander('Show more recommendations')

            with my_expander:
                for i in range(3, 10):

                    st.write(
                        '**Name:** {}\n\n**Category:** {}\n\n**Ingredients:** {}\n\n**Instructions:** {}\n\n**Serve in:** {}'
                            .format(top_items.iloc[i].Name, top_items.iloc[i].Category, top_items.iloc[i].Ingredients,
                                    top_items.iloc[i].Instructions,
                                    top_items.iloc[i].Serve_In))
                    try:
                        st.image(top_items.iloc[i].Photo_Link, width=250)
                    except:
                        st.write('Image is not available!')

                    st.write('---')


    elif cocktails_options == 'By Ingredients':

        user_input = st.text_input("what are your favorite ingredients? (gin, vodka, rum, etc.)")

        if user_input == '':
            st.write('My personal favorites are "scotch" and "whiskey"')

        else:

            topic_prob_dist = nmf_ingredients.transform(tfidf_ingredients.transform([user_input]))

            list_top_items_by_indices = list(cosine_similarity(topic_prob_dist, ingredients_matrix).argsort())[0][
                                        -1:-21:-1]

            top_items = cocktails_with_photos.iloc[list_top_items_by_indices].sample(10)

            st.write('**The top results:**')
            st.write('\n')

            for i in range(3):

                st.write(
                    '**Name:** {}\n\n**Category:** {}\n\n**Ingredients:** {}\n\n**Instructions:** {}\n\n**Serve in:** {}'
                        .format(top_items.iloc[i].Name, top_items.iloc[i].Category, top_items.iloc[i].Ingredients,
                                top_items.iloc[i].Instructions,
                                top_items.iloc[i].Serve_In))
                try:
                    st.image(top_items.iloc[i].Photo_Link, width=250)
                except:
                    st.write('Image is not available!')

                st.write('---')

            my_expander = st.beta_expander('Show more recommendations')

            with my_expander:
                for i in range(3, 10):

                    st.write(
                        '**Name:** {}\n\n**Category:** {}\n\n**Ingredients:** {}\n\n**Instructions:** {}\n\n**Serve in:** {}'
                            .format(top_items.iloc[i].Name, top_items.iloc[i].Category, top_items.iloc[i].Ingredients,
                                    top_items.iloc[i].Instructions,
                                    top_items.iloc[i].Serve_In))
                    try:
                        st.image(top_items.iloc[i].Photo_Link, width=250)
                    except:
                        st.write('Image is not available!')

                    st.write('---')
