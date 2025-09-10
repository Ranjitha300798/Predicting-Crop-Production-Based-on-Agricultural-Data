import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="🌾 Crop Production Trends & Prediction", layout="wide")
st.title("🌱 Crop Production Trends & Prediction App")

# Load Data
@st.cache_data
def load_data():
    path = r"C:\Users\Ranjitha\OneDrive\Documents\Crop_Project.csv"
    df = pd.read_csv(path, index_col=0)
    return df

df = load_data()

# Check if data loaded correctly
if df.empty:
    st.error("Data not loaded correctly.")
    st.stop()

# Data Cleaning: keep only official figures
df = df[df['flag_description'] == 'Official figure']

# Pivot the data
df_pivot = df.pivot_table(index=['area', 'item', 'year'], 
                          columns='element', 
                          values='value').reset_index()

# Drop rows with missing values in key columns
df_pivot = df_pivot.dropna(subset=['Area harvested', 'Yield', 'Production'])

# Sidebar filters
st.sidebar.header("Filter Options")
selected_country = st.sidebar.selectbox("Select Country", sorted(df_pivot['area'].unique()))
selected_crop = st.sidebar.selectbox("Select Crop", sorted(df_pivot['item'].unique()))

# Filter for selected country and crop first
country_crop_df = df_pivot[(df_pivot['area'] == selected_country) & (df_pivot['item'] == selected_crop)]

# Year filter (dynamic based on previous filters)
available_years = sorted(country_crop_df['year'].unique())
selected_year = st.sidebar.selectbox("Select Year", available_years)

# Final filtered DataFrame
filtered_df = country_crop_df[country_crop_df['year'] == selected_year]

# Show filtered table
st.subheader(f"📋 Data for {selected_crop} in {selected_country} ({selected_year})")
st.dataframe(filtered_df)

# All years data for trends
filtered_df_sorted = country_crop_df.sort_values('year')

# Visualizations in tabs
tab1, tab2, tab3 = st.tabs(["Production Trend", "Yield Trend", "Area Harvested Trend"])

with tab1:
    st.subheader(f"Production Trend: {selected_crop} in {selected_country}")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.lineplot(data=filtered_df_sorted, x='year', y='Production', marker='o', ax=ax)
    ax.set_ylabel("Production (tons)")
    st.pyplot(fig)

with tab2:
    st.subheader(f"Yield Trend: {selected_crop} in {selected_country}")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.lineplot(data=filtered_df_sorted, x='year', y='Yield', marker='o', color='green', ax=ax)
    ax.set_ylabel("Yield (kg/ha)")
    st.pyplot(fig)

with tab3:
    st.subheader(f"Area Harvested Trend: {selected_crop} in {selected_country}")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.lineplot(data=filtered_df_sorted, x='year', y='Area harvested', marker='o', color='orange', ax=ax)
    ax.set_ylabel("Area harvested (ha)")
    st.pyplot(fig)

# Prediction section in sidebar
st.sidebar.header("🔮 Predict Production")

X = df_pivot[['Area harvested', 'Yield']]
y = df_pivot['Production']

# Train Linear Regression model on a sample (to save memory)
sample_size = min(10000, len(X))
X_sample = X.sample(n=sample_size, random_state=42)
y_sample = y.loc[X_sample.index]

model = LinearRegression()
model.fit(X_sample, y_sample)

area_input = st.sidebar.number_input("Area harvested (ha):", min_value=0.0, value=1000.0)
yield_input = st.sidebar.number_input("Yield (kg/ha):", min_value=0.0, value=2000.0)

if st.sidebar.button("Predict Production"):
    pred = model.predict([[area_input, yield_input]])
    st.sidebar.success(f"Estimated Production: {pred[0]:,.2f} tons")

# Model evaluation display
with st.sidebar.expander("📊 Model Evaluation"):
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"R² Score: {r2:.3f}")
    st.write(f"Mean Absolute Error (MAE): {mae:.3f} tons")