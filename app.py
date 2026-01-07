import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(page_title="Netflix User Segmentation", layout="wide")

st.title("👥 Netflix User Segmentation (K-Means)")
st.markdown("""
This application groups Netflix users into distinct clusters based on their behaviors and demographics 
using **K-Means Clustering**.
""")

# Load dataset
@st.cache_data
def load_data():
    try:
        # Load the specific netflix dataset
        df = pd.read_csv("/home/intellact/Downloads/K_Means/netflix.csv")
        # Helper to ensure pyarrow compatibility
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
        return df
    except FileNotFoundError:
        st.error("Dataset 'netflix_cleaned_20251011_141144.csv' not found.")
        return None

df = load_data()

if df is not None:
    # Sidebar
    st.sidebar.title("Configuration")
    app_mode = st.sidebar.selectbox("Mode", ["Data Overview", "Clustering Analysis"])

    # Preprocessing
    # Select numeric columns relevant for clustering
    # Based on earlier analysis: Age, Monthly Revenue, Download Speed, etc.
    # We'll filter for numeric types automatically
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove likely irrelevant numeric cols like IDs if they exist (though select_dtypes might catch them if int)
    # Let's verify common non-feature numerics in such datasets
    ignore_cols = ['id', 'ID', 'index', 'Unnamed: 0'] 
    feature_cols = [c for c in numeric_cols if all(x not in c for x in ignore_cols)]

    if app_mode == "Data Overview":
        st.header("📊 Dataset Overview")
        st.write(f"Total Records: {len(df)}")
        st.dataframe(df.head())
        
        st.subheader("Statistics")
        st.write(df.describe())
        
        st.subheader("Correlations")
        if feature_cols:
            corr = df[feature_cols].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No numeric features found for correlation.")

    elif app_mode == "Clustering Analysis":
        st.header("🔍 K-Means Clustering")
        
        if not feature_cols:
            st.error("No numeric features available for clustering.")
        else:
            # Feature Selection
            st.subheader("1. Select Features")
            selected_features = st.multiselect("Choose features for clustering:", feature_cols, default=feature_cols[:3] if len(feature_cols)>=3 else feature_cols)
            
            if not selected_features:
                st.warning("Please select at least one feature.")
            else:
                X = df[selected_features].dropna()
                
                # Scaling
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Elbow Method
                st.subheader("2. Determine Optimal Clusters (Elbow Method)")
                wcss = []
                max_k = 10
                for i in range(1, max_k + 1):
                    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
                    kmeans.fit(X_scaled)
                    wcss.append(kmeans.inertia_)
                    
                fig_elbow, ax_elbow = plt.subplots()
                ax_elbow.plot(range(1, max_k + 1), wcss, marker='o')
                ax_elbow.set_title('Elbow Method')
                ax_elbow.set_xlabel('Number of Clusters (k)')
                ax_elbow.set_ylabel('WCSS')
                st.pyplot(fig_elbow)
                
                # Clustering
                st.subheader("3. Cluster Visualization")
                k = st.slider("Select Number of Clusters (k)", 2, 10, 3)
                
                kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                
                # Add clusters to data
                df_clustered = df.loc[X.index].copy()
                df_clustered['Cluster'] = clusters
                
                # Visualization options
                col1, col2 = st.columns(2)
                with col1:
                    x_axis = st.selectbox("X-Axis Feature", selected_features, index=0)
                with col2:
                    y_axis = st.selectbox("Y-Axis Feature", selected_features, index=1 if len(selected_features)>1 else 0)
                
                fig_cluster, ax_cluster = plt.subplots()
                sns.scatterplot(data=df_clustered, x=x_axis, y=y_axis, hue='Cluster', palette='viridis', ax=ax_cluster, s=50)
                ax_cluster.set_title(f"Clusters: {x_axis} vs {y_axis}")
                st.pyplot(fig_cluster)
                
                st.subheader("Cluster Profiles (Mean Values)")
                cluster_means = df_clustered.groupby('Cluster')[selected_features].mean()
                st.dataframe(cluster_means)
