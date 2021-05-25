# Drinks and Cocktails Recommendation

---

- In this project, I will use Natural Language Processing (NLP) to analyze texts (descriptions and reviews) to analyze over *10,000* kinds of wines, beers, liquors, and cocktails.
- I will do **topic modeling** using TF-IDF (term frequency-inverse document frequency) and NMF (Non-Negative Matrix Factorization) to seek insights data.
- Upon topics I've extracted from topic modeling, I will build the recommendation system to recommend users the most relevant products based on what they type into the system.
- **Potential impact:** to create the app to recommend drinks and cocktails to users based on their tastes and preference. Furthermore, it helps bars and restaurant owners to optimize their business.

#### Initial findings:
![](https://github.com/luongtruong77/nlp-drinks-cocktails-recommender/blob/main/figures/10_topics_by_description.png?raw=true)

- Topics can be broken down into:
    ```
    categories_by_descriptions = {
    'topic_1': 'cheap, affordable, and good value',
    'topic_2': 'whiskey, bourbon, gin, rum, tequila',
    'topic_3': 'red blend, merlot, cabernet sauvignon',
    'topic_4': 'red pinot noir',
    'topic_5': 'sweet cider',
    'topic_6': 'white chardonnay',
    'topic_7': 'beer',
    'topic_8': 'white sauvignon blanc',
    'topic_9': 'rose',
    'topic_10': 'vodka and spirits' 
    }
    ```

#### Streamlit app snapshot:
So I've put together what I've done and as a demo, as the user type in `cheap scotch`, the app would return the top 3 results (snapshot only captures 1 in this case). 
As a scotch drinker, I don't consider bourbon is the same as scotch (mainly because scotch is from Scotland and bourbon is from the USA, among other factors); however, it's close since they are both Whiskey. The problem could be the number of topics is small, so the algorithm doesn't split scotch and bourbon into 2 categories. Another possiblity is that the dataset is quite small (10,000 instances).
![](https://github.com/luongtruong77/nlp-drinks-cocktails-recommender/blob/main/figures/app_snap_shot.png?raw=true)