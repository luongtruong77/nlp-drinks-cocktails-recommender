# Spirits, Wine, Beer, and Cocktails Recommender System
#### Which drink is suitable with your personality?
---

Steven L Truong

---

## Abstract
---
In this project, I will use Natural Language Processing (NLP) with the data was acquired from multiple sources to build a system to recommend spirits, wine, beer, and cocktails to users based on their text inputs. The reviews, descriptions, and tasting information (wine and spirits), as well as ingredients (cocktails) were encoded using *Term Frequency–Inverse Document Frequency (TF-IDF)* and used to build *Non-negative Matrix Factorization (NMF)* topic modeling. Upon the topics extracted from the model, the cosine similarity metric is used to compare users input information and return the most relevant products.

The **recommender system** can be accessed [HERE](https://share.streamlit.io/luongtruong77/nlp-drinks-cocktails-recommender/main/app/app.py)

**PLEASE don't drink and drive!**

## Design
---
- Choosing our favorite drink is (almost always) not easy!
- We all have different opinions and choices when choosing our favorite alcoholic drinks. How do we pick the right one out of (possibly) over 10,000 types of drinks.
- Let's say we all come to the bar to enjoy our weekends but we don't know what to order and the bartenders are too busy to recommend personalized drinks for each one of us. How do we know what we want?
- Would it be nice if we have an app (or just web brower interface) to help us choose our suitable drink based on our preference?
- **YES!** In this project, I will answer those questions with the solution for all of us who have the same questions above.
- Possible impacts:
    - Business owners keep their customers (they will come back if they enjoy their drinks).
    - End-users will be very happy when they know what to order (good drinks).
    - Modern way to run bars and pubs.

## Data
---
There are multiple data sources:
- I scraped from [Tastings.com](https://www.tastings.com/Reviews/Latest-Spirits-Wine-Beer-Reviews.aspx) and [Caskers](https://www.caskers.com/spirits/). These sites are mainly for wine, spirits, and beer.
- I scraped from [Liquor](https://www.liquor.com/cocktail-by-spirit-4779438) and calling API from [CocktailDB](https://www.thecocktaildb.com/) for cocktails ingredients.
- The final ready-to-work-with dataset has around *12,000* instances and *24* features with 5 text-heavy features to work with (descriptions, reviews, tasting info, ingredients, and instructions).

## Algorithms
---
#### Data cleaning and features engineering
- Tokenize words, remove numbers and punctuations.
- Add more case-specific stop words to stop-words corpus.
- Combine short review and extended review into 1 feature Full_Review for exploring.

#### Recommendation System
- MDS, Isomap, t-SNE, LLE, PCA for clustering.
>**2-D t-SNE mapping**

![](https://github.com/luongtruong77/nlp-drinks-cocktails-recommender/blob/main/figures/tSNE-15.png?raw=true)
- Term Frequency–Inverse Document Frequency (TF-IDF) for encoding.
- Non-negative Matrix Factorization (NMF) for topic modeling and building recommendation system.
>**15 topics break-down**

![](https://github.com/luongtruong77/nlp-drinks-cocktails-recommender/blob/main/figures/list_of_topics.png?raw=true)

>**Recommendation system snapshots**

![](https://github.com/luongtruong77/nlp-drinks-cocktails-recommender/blob/main/figures/app_snap_shot.png?raw=true)
![](https://github.com/luongtruong77/nlp-drinks-cocktails-recommender/blob/main/figures/app_snap_shot2.png?raw=true)


## Tools
---
- Python
- Pandas
- Numpy
- Matplotlib
- Plotly
- Scikit-learn
- NLTK
- Wordcloud
- Streamlit

## Communication
- All the notebooks can be found [HERE.](https://github.com/luongtruong77/nlp-drinks-cocktails-recommender/tree/main/notebooks)
- The recommendation app can be found [HERE.](https://share.streamlit.io/luongtruong77/nlp-drinks-cocktails-recommender/main/app/app.py)
- The presentation can be found [HERE.](https://github.com/luongtruong77/nlp-drinks-cocktails-recommender/blob/main/Presentation.pdf)


