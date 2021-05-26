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

########################################


st.image(Image.open('figures/cocktails.jpg'))

st.markdown("<h1 style='text-align: center; color: black;'>Your personal drinks recommendation system!</h1>",
            unsafe_allow_html=True)
st.write('---')

########################################


options = st.sidebar.selectbox('Please choose one of the options on how to search for your drinks.',
                               ('Please drink responsibly', 'Spirits, Wine, and Beer Recommendation',
                                'Cocktails Recommendation'))

if options == 'Please drink responsibly':
    st.write("Please don't drink and drive!")

elif options == 'Spirits, Wine, and Beer Recommendation':

    choices = st.radio('Please choose how you want to be recommended by', ('By Description', 'By Tasting Info'))

    if choices == 'By Description':

        nmf = nmf_des
        tfidf = tfidf_des
        topics_matrix = topics_by_description_matrix
        topics_df = topics_by_description_df

        st.write('What would you like to drink today?')
        user_input = st.text_input(
            "Anything that you would like to describe your drink (cheap scotch, sweet wine, ipa, etc.)")

        if user_input == '':
            st.write('Please drink responsibly!')
        else:

            topic_prob_dist = nmf.transform(tfidf.transform([user_input]))

            list_top_items_by_indices = list(cosine_similarity(topic_prob_dist, topics_matrix).argsort())[0][
                                        -1:-31:-1]

            top_items = topics_df.iloc[list_top_items_by_indices].sample(10)


            st.write('**The top results:**')
            st.write('\n')
            for i in range(3):
                st.write('**Name:** {}\n\n**Country:** {}\n\n**Alcohol Volume:** {}\n\n**Aroma:** {}\n\n**Flavor:** {}\n\n'
                         '**Price:** ${}\n\n**Comments:** {}\n\n*{}*'.
                         format(top_items.iloc[i].Name, top_items.iloc[i].Country, top_items.iloc[i].Alcohol_Vol,
                                top_items.iloc[i].Aroma, top_items.iloc[i].Flavor, top_items.iloc[i].Price,
                                top_items.iloc[i].Bottom_Line, top_items.iloc[i].Review))
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
                        '**Price:** ${}\n\n**Comments:** {}\n\n*{}*'.
                            format(top_items.iloc[i].Name, top_items.iloc[i].Country, top_items.iloc[i].Alcohol_Vol,
                                   top_items.iloc[i].Aroma, top_items.iloc[i].Flavor, top_items.iloc[i].Price,
                                   top_items.iloc[i].Bottom_Line, top_items.iloc[i].Review))
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
                    st.write(
                        '**Name:** {}\n\n**Country:** {}\n\n**Alcohol Volume:** {}\n\n**Style:** {}\n\n**Flavor:** {}\n\n'
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

    cocktails_options = st.radio('Please choose how you want to search:', ('By Name', 'By Ingredients'))

    if cocktails_options == 'By Name':

        user_input = st.text_input("case insensitive (gin and tonic, old-fashioned, etc.)")

        if user_input == '':
            st.write('Cheers!')

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
            st.write('Cheers!')

        else:

            topic_prob_dist = nmf_ingredients.transform(tfidf_ingredients.transform([user_input]))

            list_top_items_by_indices = list(cosine_similarity(topic_prob_dist, ingredients_matrix).argsort())[0][-1:-21:-1]

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
