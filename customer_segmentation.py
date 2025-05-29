import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('customer_segmentation_output.csv')

# Sidebar
st.sidebar.title("Filters")
selected_cluster = st.sidebar.selectbox("Choose Cluster", sorted(df['clusters'].unique()))

# Title
st.title("Customer Segmentation Dashboard")

# 1. Cluster size
st.subheader("Cluster Distribution")
cluster_counts = df['clusters'].value_counts().sort_index()
st.bar_chart(cluster_counts)

# 2. Summary metrics for selected cluster
st.subheader(f" Summary for Cluster {selected_cluster}")
filtered = df[df['clusters'] == selected_cluster]

st.metric("Customers in Cluster", len(filtered))
st.metric("Avg Order Value", round(filtered['avg_order_value'].mean(), 2))
st.metric("Frequency", round(filtered['purchase_frequency'].mean(), 2))
st.metric("Avg Review Score", round(filtered['avg_review_score'].mean(), 2))

# 3. Boxplot comparison
st.subheader(" Feature Distributions by Cluster")
features = ['avg_order_value', 'purchase_frequency', 'num_categories_bought', 'avg_review_score']

for feat in features:
    fig, ax = plt.subplots()
    sns.boxplot(x='clusters', y=feat, data=df, palette='Set2', ax=ax)
    st.pyplot(fig)

# 4. Customers by state and cluster
st.subheader(" Customer Distribution by State and Cluster")
state_counts = df.groupby(['customer_state', 'clusters']).size().unstack(fill_value=0)
st.dataframe(state_counts)

# 5. PCA scatter plot
if 'pc1' in df.columns and 'pc2' in df.columns:
    st.subheader(" 2D PCA of Clusters")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='pc1', y='pc2', hue='clusters', palette='Set1', ax=ax)
    st.pyplot(fig)
