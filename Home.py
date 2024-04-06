import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px


st.markdown("<h1 style='text-align: center;'>SISTEM REKOMENDASI FILM</h1>", unsafe_allow_html=True)
st.image('movie.webp') 

st.markdown("""
Proyek ini bertujuan untuk mengembangkan Sistem Rekomendasi Film yang mengutamakan kesamaan genre. Sistem ini menggunakan metode content-based filtering untuk memberikan rekomendasi film yang sesuai dengan preferensi masing-masing pengguna.

Pada Content-based filtering ini kita melakukan pendekatan dengan menganalisis fitur-fitur dari item yang sudah dikenal oleh pengguna. Berdasarkan atribut tersebut, sistem akan menemukan kesamaan antara film yang telah disukai oleh pengguna dengan film lainnya, lalu merekomendasikan film-film yang memiliki atribut serupa.

Dalam konteks ini, sistem membagi film ke dalam kelompok-kelompok atau kluster yang memiliki karakteristik serupa. Dengan pendekatan ini, sistem dapat memberikan rekomendasi yang lebih akurat dan personal karena berdasarkan pada preferensi dan kesukaan pengguna.
""")

