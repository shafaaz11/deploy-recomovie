import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px

st.header('Modelling (Unsupervised)')

def load_data(nrows):
    data = 'data_encoded.csv'
    df = pd.read_csv(data, nrows=nrows)
    return df
df = load_data(1000)
st.markdown("---")

st.sidebar.header('Pilih Film')
option = st.sidebar.selectbox('Judul Film', df['original_title'])

# Pembagian data menjadi data latih dan data uji
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

X_train_numeric = X_train.drop('original_title', axis=1)

silhouette_scores = []
for n_clusters in range(2, 11):
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = hc.fit_predict(X_train_numeric)
    silhouette_avg = silhouette_score(X_train_numeric, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Visualisasi Silhouette Score
st.subheader('Silhouette Score')
fig_silhouette = px.line(x=range(2, 11), y=silhouette_scores, markers=True)
fig_silhouette.update_layout(xaxis_title='Jumlah Kluster', yaxis_title='Silhouette Score')
st.plotly_chart(fig_silhouette)
st.write("Grafik di atas menunjukkan Silhouette Score untuk berbagai jumlah kluster. Silhouette Score mengukur seberapa mirip sebuah objek dengan kluster sendiri (kohesi) dibandingkan dengan kluster lain (pemisahan). Skor Silhouette yang lebih tinggi menunjukkan pembentukan kluster yang lebih baik.")
st.markdown("---")

st.subheader('Model Agglomerative Clustering')

# Buat objek model HAC dengan jumlah cluster yang dipilih
n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
cluster_labels = hc.fit_predict(X_train_numeric)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_numeric)

import plotly.express as px

# Visualisasi PCA
scatter_data = pd.DataFrame({
    'Komponen Utama 1': X_train_pca[:, 0],
    'Komponen Utama 2': X_train_pca[:, 1],
    'Kluster': cluster_labels
})

fig = px.scatter(scatter_data, x='Komponen Utama 1', y='Komponen Utama 2', color='Kluster', 
                 title='Visualisasi Clustering Agglomerative', 
                 labels={'Komponen Utama 1': 'Komponen Utama 1', 'Komponen Utama 2': 'Komponen Utama 2'}, 
                 hover_name=scatter_data.index)

st.plotly_chart(fig)

st.write("Diagram di atas memvisualisasikan hasil pengelompokan menggunakan PCA. Setiap titik mewakili sebuah film, diwarnai berdasarkan kluster mereka.")
st.markdown("---")


if st.sidebar.button('Rekomendasikan film serupa'):
    
    # Fungsi untuk merekomendasikan film dari klaster yang sama
    def recommend_similar_movies(input_title, data_encoded, cluster_labels, n_recommendations=5):
        input_movie_index = data_encoded.index[data_encoded['original_title'] == input_title][0]
        input_movie_cluster = cluster_labels[input_movie_index]
        similar_movies_indices = [i for i, label in enumerate(cluster_labels) if label == input_movie_cluster and i != input_movie_index]
        similar_movies = data_encoded.iloc[similar_movies_indices]

        recommended_movies = similar_movies.sample(n=n_recommendations, random_state=42)
        recommended_titles = recommended_movies['original_title'].tolist()

        return recommended_titles

    recommended_movies = recommend_similar_movies(option, df, cluster_labels)
    for movie in recommended_movies:
        st.sidebar.write(movie)

st.subheader('Nilai Metrik Model Agglomerative')

# menampilkan visualisasi metric
metrics_output1 = [202.593785, 2.142428, 0.15072]
metrics_output2 = [87.606599, 2.823956, 0.09435]

metric_names = ['Calinski Harabasz', 'Davies Bouldin', 'Silhouette']
n_values = ['n=3', 'n=8']

for i in range(len(metric_names)):
    st.metric(label=metric_names[i], value=metrics_output1[i], delta=metrics_output2[i], delta_color="inverse")
st.write("Diatas menunjukan perbandingan nilai metrik dengan cluster yang berbeda, dimana kita menggunakan score dengan model n terbaik")
