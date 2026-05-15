import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Real Estate Analytics Platform",
    layout="wide"
)

# ---------------- CUSTOM DARK THEME ----------------

st.markdown("""
<style>

.stApp {
    background-color: #0b0f0b;
    color: #00ff88;
}

h1, h2, h3, h4, h5 {
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

st.sidebar.title("AI Property Intelligence")

st.sidebar.write("Advanced Real Estate Analytics System")

st.sidebar.markdown("---")

st.sidebar.subheader("User Requirements")

preferred_location = st.sidebar.selectbox(
    "Select Preferred Location",
    [
        "Mumbai",
        "Pune",
        "Bangalore",
        "Hyderabad",
        "Delhi",
        "Chennai"
    ]
)

budget = st.sidebar.slider(
    "Select Budget",
    100000,
    1000000,
    500000
)

st.sidebar.markdown("---")

# ---------------- DATASET ----------------

data = {
    "Location": [
        "Mumbai","Mumbai",
        "Pune","Pune",
        "Bangalore","Bangalore",
        "Hyderabad","Delhi",
        "Chennai","Mumbai"
    ],

    "Area": [
        1000,1500,1800,2200,
        2500,3000,3500,4000,
        2800,4500
    ],

    "Bedrooms": [
        2,3,3,4,4,
        5,5,6,4,6
    ],

    "Bathrooms": [
        1,2,2,3,3,
        4,4,5,3,5
    ],

    "Age": [
        20,15,12,10,
        8,6,5,3,7,2
    ],

    "Garage": [
        1,1,2,2,
        2,3,3,4,
        2,4
    ],

    "Price": [
        200000,320000,380000,
        450000,520000,620000,
        720000,850000,580000,
        950000
    ]
}

df = pd.DataFrame(data)

# ---------------- FILTER USER NEEDS ----------------

filtered_houses = df[
    (df["Location"] == preferred_location) &
    (df["Price"] <= budget)
]

# ---------------- MACHINE LEARNING ----------------

X = df[["Area","Bedrooms","Bathrooms","Age","Garage"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

# ---------------- HEADER ----------------

st.title("AI Powered Real Estate Analytics Platform")

st.write(
    "Enterprise-level Machine Learning dashboard for intelligent property analysis and prediction."
)

st.markdown("---")

# ---------------- KPI SECTION ----------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Properties",
        len(df)
    )

with col2:
    st.metric(
        "Average Market Price",
        f"${df['Price'].mean():,.0f}"
    )

with col3:
    st.metric(
        "Maximum Property Value",
        f"${df['Price'].max():,.0f}"
    )

with col4:
    st.metric(
        "Model Accuracy",
        f"{r2*100:.2f}%"
    )

st.markdown("---")

# ---------------- USER INPUT ----------------

st.subheader("Live Property Prediction")

c1, c2, c3 = st.columns(3)

with c1:
    area = st.slider(
        "Area (sq ft)",
        500,
        5000,
        2000
    )

with c2:
    bedrooms = st.slider(
        "Bedrooms",
        1,
        8,
        3
    )

with c3:
    bathrooms = st.slider(
        "Bathrooms",
        1,
        6,
        2
    )

c4, c5 = st.columns(2)

with c4:
    age = st.slider(
        "Property Age",
        1,
        30,
        10
    )

with c5:
    garage = st.slider(
        "Garage Capacity",
        0,
        4,
        2
    )

# ---------------- LIVE PREDICTION ----------------

input_data = np.array([
    [
        area,
        bedrooms,
        bathrooms,
        age,
        garage
    ]
])

predicted_price = model.predict(input_data)[0]

st.success(
    f"Predicted Property Price: ${predicted_price:,.2f}"
)

# ---------------- LIVE DYNAMIC DATA ----------------

dynamic_df = pd.DataFrame({
    "Features": [
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Age",
        "Garage"
    ],

    "Values": [
        area,
        bedrooms,
        bathrooms,
        age,
        garage
    ]
})

st.markdown("---")

# ---------------- LIVE CHARTS ----------------

st.subheader("Real-Time Property Analytics")

chart1, chart2 = st.columns(2)

# Dynamic Bar Chart

with chart1:

    fig1, ax1 = plt.subplots(figsize=(6,4))

    ax1.bar(
        dynamic_df["Features"],
        dynamic_df["Values"]
    )

    ax1.set_facecolor("#111111")

    fig1.patch.set_facecolor("#111111")

    ax1.tick_params(colors='#00ff88')

    ax1.set_title(
        "Property Feature Distribution",
        color="#00ff88"
    )

    st.pyplot(fig1)

# Dynamic Prediction Graph

with chart2:

    live_prices = [
        predicted_price * 0.8,
        predicted_price * 0.9,
        predicted_price,
        predicted_price * 1.1,
        predicted_price * 1.2
    ]

    fig2, ax2 = plt.subplots(figsize=(6,4))

    ax2.plot(
        live_prices,
        linewidth=3
    )

    ax2.set_facecolor("#111111")

    fig2.patch.set_facecolor("#111111")

    ax2.tick_params(colors='#00ff88')

    ax2.set_title(
        "Predicted Market Trend",
        color="#00ff88"
    )

    st.pyplot(fig2)

# ---------------- PROPERTY DATABASE ----------------

st.markdown("---")

st.subheader("Property Database")

st.dataframe(
    df,
    use_container_width=True
)

# ---------------- USER NEEDS ----------------

st.markdown("---")

st.subheader("Recommended Properties Based on User Needs")

if len(filtered_houses) > 0:

    st.dataframe(
        filtered_houses,
        use_container_width=True
    )

else:

    st.warning(
        "No properties available for selected location and budget."
    )

# ---------------- ADVANCED ANALYTICS ----------------

st.markdown("---")

st.subheader("Advanced Market Analytics")

a1, a2 = st.columns(2)

# Scatter Plot

with a1:

    fig3, ax3 = plt.subplots(figsize=(6,4))

    ax3.scatter(
        df["Area"],
        df["Price"]
    )

    ax3.set_facecolor("#111111")

    fig3.patch.set_facecolor("#111111")

    ax3.tick_params(colors='#00ff88')

    ax3.set_title(
        "Area vs Price Analysis",
        color="#00ff88"
    )

    st.pyplot(fig3)

# Line Trend

with a2:

    fig4, ax4 = plt.subplots(figsize=(6,4))

    ax4.plot(
        df["Price"],
        linewidth=3
    )

    ax4.set_facecolor("#111111")

    fig4.patch.set_facecolor("#111111")

    ax4.tick_params(colors='#00ff88')

    ax4.set_title(
        "Market Growth Trend",
        color="#00ff88"
    )

    st.pyplot(fig4)

# ---------------- HEATMAP ----------------

st.markdown("---")

st.subheader("Correlation Heatmap")

correlation = df.drop(columns=["Location"]).corr()

fig5, ax5 = plt.subplots(figsize=(8,5))

heatmap = ax5.imshow(
    correlation,
    cmap="Greens"
)

ax5.set_xticks(
    range(len(correlation.columns))
)

ax5.set_yticks(
    range(len(correlation.columns))
)

ax5.set_xticklabels(
    correlation.columns,
    rotation=45,
    color="#00ff88"
)

ax5.set_yticklabels(
    correlation.columns,
    color="#00ff88"
)

fig5.patch.set_facecolor("#111111")

ax5.set_facecolor("#111111")

plt.colorbar(heatmap)

st.pyplot(fig5)

# ---------------- MODEL PERFORMANCE ----------------

st.markdown("---")

st.subheader("Machine Learning Model Performance")

m1, m2 = st.columns(2)

with m1:
    st.metric(
        "Mean Absolute Error",
        f"{mae:.2f}"
    )

with m2:
    st.metric(
        "R² Score",
        f"{r2:.4f}"
    )

# ---------------- FOOTER ----------------

st.markdown("---")

st.write(
    "Developed using Python, Streamlit, Scikit-Learn, Pandas, NumPy, and Matplotlib."
)