import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px
import streamlit as st


def load_data(nrows):
    data = 'movie.csv'
    df = pd.read_csv(data, nrows=nrows)
    return df
df = load_data(1000)

st.header('Exploratory Data Analysis')

st.markdown("---")
def genre_name(genre_list):
    return [genre["name"] for genre in eval(genre_list)]


df["genre_name"] = df["genres"].apply(genre_name)
genre_counts = df["genre_name"].explode().value_counts()
top_genres = genre_counts.head(5)


st.sidebar.subheader('Genre Film Terpopuler')
selected_genre = st.sidebar.selectbox("Pilih Genre", top_genres.index.tolist())
selected_genre_count = genre_counts[selected_genre]
st.sidebar.write(f"Jumlah film dengan genre {selected_genre}: {selected_genre_count}")

# Visualisasi Top Genre
st.subheader('Top Genre')
fig_donut = px.pie(names=top_genres.index.tolist(), values=top_genres.values.tolist(), hole=0.2)
fig_donut.update_traces(textposition='inside', textinfo='percent+label')
fig_donut.update_layout(width=460, height=460)
st.plotly_chart(fig_donut)
st.write("Grafik donat di atas menampilkan komposisi genre terpopuler dalam dataset film. Setiap bagian donat mewakili persentase dari total jumlah film dalam masing-masing genre.")
st.markdown("---")

# Visualisasi Film berdasarkan Genre
st.subheader('Jumlah Film Berdasarkan Genre')
st.bar_chart(genre_counts)
st.write("Grafik bar di atas menampilkan jumlah film dari setiap genre. Pada Grafik Genre Drama mendominasi jumlah film dimana menunjukkan popularitas yang tinggi di antara genre lainnya.")
st.markdown("---")

# Visualisasi Rating
st.subheader('Distribusi Rating Film')
rating_counts = df['vote_average'].value_counts().sort_index()
st.bar_chart(rating_counts)
st.write("Grafik bar di atas menampilkan banyaknya jumlah untuk setiap rating. Pada grafik menunjukkan bahwa sebagian besar film memiliki peringkat di sekitar 5 hingga 7.5, dengan puncak distribusi terletak di sekitar nilai 6. Ini menunjukkan bahwa mayoritas film cenderung mendapatkan peringkat yang relatif positif.")
st.markdown("---")

# Visualisasi Popularitas
def get_top_10_movies(df, column):
    top_10 = df.nlargest(10, column)
    top_10_sorted = top_10.sort_values(by=column, ascending=True) 
    st.subheader(f'Top 10 Film Berdasarkan {column}')
    fig = px.bar(top_10_sorted, x=column, y='original_title', orientation='h')
    fig.update_xaxes(title='Popularity')
    fig.update_yaxes(title='Judul Film')
    st.plotly_chart(fig)

get_top_10_movies(df, "popularity")
st.write("Grafik batang di atas menampilkan 10 film terpopuler berdasarkan popularitasnya.")
st.markdown("---")

# Visualisasi Perbandingan: Jumlah Film Berdasarkan Tahun Rilis
st.subheader('Jumlah Film Berdasarkan Tahun Rilis')
year_counts = df['release_date'].str.split('-').str[0].value_counts().sort_index()
st.bar_chart(year_counts)
st.write("Grafik batang di atas menampilkan jumlah film berdasarkan tahun rilisnya, memungkinkan kita untuk membandingkan jumlah film yang dirilis setiap tahunnya.")

# Visualisasi Hubungan: Hubungan antara Rating dan Popularitas
st.subheader('Hubungan antara Rating dan Popularitas')
fig = px.scatter(df, x='vote_average', y='popularity', hover_name='original_title')
fig.update_layout(xaxis_title='Rating', yaxis_title='Popularitas')
st.plotly_chart(fig)
st.write("Scatter plot di atas menampilkan hubungan antara rating film dan popularitasnya. Dengan melihat plot ini, kita dapat melihat apakah film dengan rating yang lebih tinggi cenderung lebih populer.")
