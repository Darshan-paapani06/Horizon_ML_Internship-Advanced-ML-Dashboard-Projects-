import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Movie Recommendation Platform",
    layout="wide"
)

# ---------------- CUSTOM DARK GREEN THEME ----------------

st.markdown("""
<style>

.stApp {
    background-color: #0b0f0b;
    color: #00ff88;
}

h1, h2, h3, h4 {
    color: #00ff88;
}

[data-testid="stSidebar"] {
    background-color: #111111;
}

[data-testid="metric-container"] {
    background-color: #161616;
    border: 1px solid #00ff88;
    padding: 15px;
    border-radius: 12px;
}

.stButton>button {
    background-color: #00ff88;
    color: black;
    border-radius: 10px;
    border: none;
    font-weight: bold;
}

.stDataFrame {
    border: 1px solid #00ff88;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.title("AI Movie Intelligence")

st.sidebar.write("Advanced Recommendation Engine")

st.sidebar.markdown("---")

genre_filter = st.sidebar.selectbox(
    "Preferred Genre",
    [
        "All",
        "Sci-Fi",
        "Action",
        "Drama",
        "Thriller",
        "Romance",
        "Fantasy"
    ]
)

minimum_rating = st.sidebar.slider(
    "Minimum Rating",
    1.0,
    10.0,
    7.0
)

minimum_year = st.sidebar.slider(
    "Release Year",
    2000,
    2025,
    2010
)

st.sidebar.markdown("---")

# ---------------- DATASET ----------------

movies = pd.DataFrame({

    "title": [
        "Inception",
        "Interstellar",
        "The Dark Knight",
        "Avatar",
        "Titanic",
        "Avengers Endgame",
        "Doctor Strange",
        "Iron Man",
        "The Matrix",
        "John Wick",
        "Joker",
        "Oppenheimer"
    ],

    "genre": [
        "Sci-Fi Thriller",
        "Sci-Fi Drama",
        "Action Thriller",
        "Sci-Fi Fantasy",
        "Romance Drama",
        "Action Sci-Fi",
        "Fantasy Action",
        "Action Technology",
        "Sci-Fi Action",
        "Action Thriller",
        "Drama Thriller",
        "Drama Biography"
    ],

    "rating": [
        8.8,
        8.7,
        9.0,
        7.9,
        7.8,
        8.4,
        7.6,
        7.9,
        8.7,
        7.5,
        8.4,
        8.6
    ],

    "year": [
        2010,
        2014,
        2008,
        2009,
        1997,
        2019,
        2016,
        2008,
        1999,
        2014,
        2019,
        2023
    ],

    "duration": [
        148,
        169,
        152,
        162,
        195,
        181,
        115,
        126,
        136,
        101,
        122,
        180
    ]
})

# ---------------- FILTERING ----------------

filtered_movies = movies[
    (movies["rating"] >= minimum_rating) &
    (movies["year"] >= minimum_year)
]

if genre_filter != "All":
    filtered_movies = filtered_movies[
        filtered_movies["genre"].str.contains(genre_filter)
    ]

# ---------------- VECTORIZATION ----------------

cv = CountVectorizer()

matrix = cv.fit_transform(movies["genre"])

similarity = cosine_similarity(matrix)

# ---------------- RECOMMENDATION FUNCTION ----------------

def recommend(movie):

    index = movies[movies["title"] == movie].index[0]

    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []

    for i in movie_list:
        recommendations.append(
            movies.iloc[i[0]].title
        )

    return recommendations

# ---------------- HEADER ----------------

st.title("AI Powered Movie Recommendation Platform")

st.write(
    "Enterprise-level intelligent recommendation engine using Machine Learning and similarity analytics."
)

st.markdown("---")

# ---------------- KPI METRICS ----------------

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Total Movies",
        len(movies)
    )

with m2:
    st.metric(
        "Average Rating",
        round(movies["rating"].mean(), 2)
    )

with m3:
    st.metric(
        "Highest Rating",
        movies["rating"].max()
    )

with m4:
    st.metric(
        "Latest Movie",
        movies["year"].max()
    )

st.markdown("---")

# ---------------- USER INPUT ----------------

st.subheader("Movie Recommendation Engine")

selected_movie = st.selectbox(
    "Select a Movie",
    movies["title"].values
)

# ---------------- LIVE RECOMMENDATIONS ----------------

recommended_movies = recommend(selected_movie)

st.success(
    f"Top Recommendations based on {selected_movie}"
)

recommendation_df = pd.DataFrame({
    "Recommended Movies": recommended_movies
})

st.dataframe(
    recommendation_df,
    use_container_width=True
)

# ---------------- MOVIE ANALYTICS ----------------

st.markdown("---")

st.subheader("Real-Time Movie Analytics")

c1, c2 = st.columns(2)

# Ratings Chart

with c1:

    fig1, ax1 = plt.subplots(figsize=(6,4))

    ax1.bar(
        movies["title"],
        movies["rating"]
    )

    ax1.set_facecolor("#111111")

    fig1.patch.set_facecolor("#111111")

    ax1.tick_params(
        colors='#00ff88',
        rotation=90
    )

    ax1.set_title(
        "Movie Ratings Analysis",
        color="#00ff88"
    )

    st.pyplot(fig1)

# Duration Chart

with c2:

    fig2, ax2 = plt.subplots(figsize=(6,4))

    ax2.plot(
        movies["duration"],
        linewidth=3
    )

    ax2.set_facecolor("#111111")

    fig2.patch.set_facecolor("#111111")

    ax2.tick_params(colors='#00ff88')

    ax2.set_title(
        "Movie Duration Trend",
        color="#00ff88"
    )

    st.pyplot(fig2)

# ---------------- LIVE GENRE DISTRIBUTION ----------------

st.markdown("---")

st.subheader("Genre Distribution Analytics")

genre_counts = movies["genre"].value_counts()

fig3, ax3 = plt.subplots(figsize=(8,5))

ax3.pie(
    genre_counts,
    labels=genre_counts.index,
    autopct="%1.1f%%"
)

fig3.patch.set_facecolor("#111111")

st.pyplot(fig3)

# ---------------- USER NEEDS SECTION ----------------

st.markdown("---")

st.subheader("Movies Matching User Preferences")

if len(filtered_movies) > 0:

    st.dataframe(
        filtered_movies,
        use_container_width=True
    )

else:

    st.warning(
        "No movies found for selected filters."
    )

# ---------------- ADVANCED ANALYTICS ----------------

st.markdown("---")

st.subheader("Advanced Recommendation Analytics")

a1, a2 = st.columns(2)

# Rating vs Year

with a1:

    fig4, ax4 = plt.subplots(figsize=(6,4))

    ax4.scatter(
        movies["year"],
        movies["rating"]
    )

    ax4.set_facecolor("#111111")

    fig4.patch.set_facecolor("#111111")

    ax4.tick_params(colors='#00ff88')

    ax4.set_title(
        "Rating vs Release Year",
        color="#00ff88"
    )

    st.pyplot(fig4)

# Recommendation Score Simulation

with a2:

    simulated_scores = np.random.randint(
        70,
        100,
        size=10
    )

    fig5, ax5 = plt.subplots(figsize=(6,4))

    ax5.plot(
        simulated_scores,
        linewidth=3
    )

    ax5.set_facecolor("#111111")

    fig5.patch.set_facecolor("#111111")

    ax5.tick_params(colors='#00ff88')

    ax5.set_title(
        "Recommendation Confidence Trend",
        color="#00ff88"
    )

    st.pyplot(fig5)

# ---------------- HEATMAP ----------------

st.markdown("---")

st.subheader("Movie Feature Correlation Heatmap")

heatmap_data = movies[[
    "rating",
    "year",
    "duration"
]].corr()

fig6, ax6 = plt.subplots(figsize=(7,5))

heatmap = ax6.imshow(
    heatmap_data,
    cmap="Greens"
)

ax6.set_xticks(
    range(len(heatmap_data.columns))
)

ax6.set_yticks(
    range(len(heatmap_data.columns))
)

ax6.set_xticklabels(
    heatmap_data.columns,
    color="#00ff88"
)

ax6.set_yticklabels(
    heatmap_data.columns,
    color="#00ff88"
)

fig6.patch.set_facecolor("#111111")

ax6.set_facecolor("#111111")

plt.colorbar(heatmap)

st.pyplot(fig6)

# ---------------- LIVE TREND ANALYSIS ----------------

st.markdown("---")

st.subheader("Movie Trend Analysis")

trend_df = pd.DataFrame({
    "Year": movies["year"],
    "Rating": movies["rating"]
})

trend_df = trend_df.sort_values("Year")

st.line_chart(
    trend_df,
    x="Year",
    y="Rating"
)



st.markdown("---")

st.write(
    "Developed using Python, Streamlit, Pandas, NumPy, Matplotlib, and Scikit-Learn."
)