# Spirits, Wine, Beer, and Cocktails Recommender System
#### Which drink is suitable with your personality?
---

Steven L Truong

---
#### Choosing our favorite drink is (almost always) not easy!
- We all have different opinions and choices when choosing our favorite alcoholic drinks. How do we pick the right one out of (possibly) over **10,000** types of drinks.
- Let's say we all come to the bar to enjoy our weekends but we don't know what to order and the bartenders are too busy to recommend personalized drinks for each one of us. How do we know what we want?
- Would it be nice if we have an app (or just web brower interface) to help us choose our suitable drink based on our preference?

#### What is this project about?
- [Ba Bar Lounge - Seattle](https://babarseattle.com/cold-drink/) is a Vietnamese restaurant that serves street food and cold drinks, and they want to improve their customers' experience by recommending them their favorite drinks.
- In this project, I will build the recommender system to help users personalize their drinks based on their input information (such as **flavor** or **sweetness**).
- This project will potentially help the bussiness keep customers (they will come back if they enjoy their drinks) and the users will be happy when they are not struggling on deciding which drink they should order.

#### Task:
The task is to do sentiment analysis with text review and build recommender system.

#### Data:
- Over **10,000** data points are scraped from [tastings.com](https://www.tastings.com/Home.aspx) containing each drink's attributes (description, flavor, pairing, price, etc) and their reviews from critics. 
- Over **600** more data points are acquired using [TheCocktailDB API](https://www.thecocktaildb.com/api.php) containing photos and ingredients.

#### Algorithms:
I am planing to use:
- Topic Modeling (Latent Dirichlet Allocation - LDA)
- Latent Semantic Analysis (LSA)
- Non-Negative Matrix Factorization (NMF)

#### Tools:
Tools I intend to use:
- Python
- Pandas
- Numpy
- NLTK
- Scikit-learn

If time permitted, I would like to deploy my model using:
- Streamlit
- Heroku

#### MVP:
- Baseline sentiment analysis and topic modeling based on drinks' descriptions and reviews.
- Recommend top 5 drinks based on the information users input.