import pandas as pd
import streamlit as st
import numpy as np

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output


# testing library kmeans : 
from sklearn.cluster import KMeans

data = pd.read_excel('diabetes.xlsx', sheet_name='diabetes', header=0)
feature = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI","Age"]

data_select = data[feature]




# fungsi standarisasi data 
def standarized_data (data):
  standarized = (data - data.mean() / data.std())
  return standarized

# standarized data yang digunakan untuk penentuan centroid dll ... 
data_standarized = standarized_data(data_select)

# def function  random_centroids fungsi menentukan random centroids
def random_centroids(data, k):
  centroids = []
  for i in range (k):
    centroid = data_standarized.apply(lambda x: float(x.sample()))
    st.write("random centroid untuk klaster ke -",i,":",centroid)
    centroids.append(centroid)
  st.write("Hasil centroid untuk masing - masing klaster :")
  result =  pd.concat(centroids, axis = 1 )
  return result


# def get labels distancte fungsi penentuan jarak dan pemberian label 
def get_labels (data,centroids):
  distances = centroids.apply(lambda x: np.sqrt(((data - x) ** 2 ).sum(axis = 1)))
  st.write("**Jarak Item & Label Klaster**")
  st.write(distances)
  return distances.idxmin(axis =1 )

# def new centroid 
def new_centroids(data, labels, k):
  param = data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T
  st.write("New Centroid : ",param)
  return   data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T




# display data 
def plot_clusters(data, labels, centroids, iteration):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids.T)
    
    plt.figure()
    plt.title(f'Iteration {iteration}')
    plt.scatter(x=data_2d[:, 0], y=data_2d[:, 1], c=labels)
    plt.scatter(x=centroids_2d[:, 0], y=centroids_2d[:, 1], marker='x', color='red')
    st.pyplot(plt)

st.title("Kelompok Algoritma K-Means Kasus Diabetes AISD-7:")
st.write("")

col1, col2 = st.columns([1,2])

with col1: 
  st.write("**NIM**")
  st.write("10122269")
  st.write("10122256")
  st.write("10122265")
  st.write("10122510")

with col2: 
  st.write("**Nama**")
  st.write("Erwin Hafiz Triadi")
  st.write("Muhammad Farid Nurrahman ")
  st.write("Muhammad Pradipta Waskitha")
  st.write("Fikkry Ihza Fachrezi")



st.title("Penerapan algoritma K-Means pada dataset diabetes.xlsx")

tab1, tab2 = st.tabs(["Static Kmeans", "Customize Kluster"])



with tab1:
  st.subheader("Data Diabetes")
  st.write(data)

  st.subheader("Feature yang digunakan untuk clustering")
  st.write(feature)

  st.write(data_select)
  st.write("Gambaran awal dataset diabetes")
  st.write(data_select.describe())



  st.subheader("Tahapan yang akan dilakukan dalam penerapan K-Means")



  st.write("Tahapan Penerapan K-Means: ")
  st.write("1. Standarisasi data")
  st.write("2. Inisialisasi random centroid berdasarkan kolom masing - masing dan penentuan jumlah klaster")
  st.write("3. Penentuan jarak dan label klaster")
  st.write("4. Perubahan titik tengah dari centroid baru sampai sama dengan centroid lama")
  st.write("5. Repeat 3 - 4")

  st.write("**Lets Start**")
  st.subheader("Tahapan 1: Standarisasi data ")
  st.write(data_standarized)
  st.write("Gambaran awal dari data standarisasi")
  st.write(data_standarized.describe())



  st.subheader("Tahapan 2: Inisialisasi random centroid berdasarkan kolom masing - masing dan penentuan klaster")
  st.write(" Klaster = 3 ")
  centroids = random_centroids(data_standarized, 3)
  st.write("Centroid : ", centroids)

  st.subheader("Tahapan 3: Penentuan jarak dan label")
  labels = get_labels(data_standarized,centroids)
  
  st.write(labels)
  st.write(labels.value_counts())

  st.subheader("Tahapan 4: Perubahan titik tengah dari centroid baru sampai sama dengan centroid lama")
  max_iteration  = 100
  k = 3

  centroids = random_centroids(data_standarized, k)
  
  old_centroids = pd.DataFrame()
  iteration = 1
  # st.write(old_centroids)

  while iteration < max_iteration and not centroids.equals(old_centroids):
    st.write(f"Iterasi: {iteration}")
    # st.write("Centroid in iteration : " ,centroids)
    old_centroids = centroids
    st.write("Old centroid sebagai threshold", old_centroids)

    labels= get_labels(data_standarized, centroids)
    centroids = new_centroids(data_standarized, labels , k )

    plot_clusters(data_standarized, labels, centroids, iteration)
    iteration += 1

  st.write(old_centroids)

  st.write("Proses clustering selesai.")
  st.pyplot(plot_clusters(data_standarized, labels, centroids, iteration))

  st.subheader("hasil cluster akhir ")
  st.write(centroids)
    
  st.write("Perbandingan jika menggunakan library kmeans")
  kmeans = KMeans(3)
  kmeans.fit(data_standarized)
  centroids = kmeans.cluster_centers_

  # st.write(centroids)
  st.write(pd.DataFrame(centroids,columns=feature).T)
  

with tab2:
    st.subheader("Masukkan jumlah klaster yang diinginkan")
    k = st.number_input("Jumlah klaster", min_value=1, value=3, step=1)

    st.write("Jumlah klaster yang dipilih:", k)

    if st.button("Mulai K-Means Clustering"):
        st.subheader("Tahapan 1: Standarisasi data ")
        st.write(data_standarized)
        st.write("Gambaran awal dari data standarisasi")
        st.write(data_standarized.describe())

        st.subheader(f"Tahapan 2: Inisialisasi random centroid berdasarkan kolom masing - masing dan penentuan klaster = {k}")
        centroids = random_centroids(data_standarized, k)
        st.write(centroids)

        st.subheader("Tahapan 3: Penentuan jarak dan label")
        labels = get_labels(data_standarized, centroids)
        st.write(labels)
        st.write(labels.value_counts())

        st.subheader("Tahapan 4: Perubahan titik tengah dari centroid baru sampai sama dengan centroid lama")
        max_iteration = 100
        old_centroids = pd.DataFrame()
        iteration = 1

        while iteration < max_iteration and not centroids.equals(old_centroids):
            st.write(f"Iterasi: {iteration}")
            old_centroids = centroids

            labels = get_labels(data_standarized, centroids)
            centroids = new_centroids(data_standarized, labels, k)

            plot_clusters(data_standarized, labels, centroids, iteration)
            iteration += 1

        st.write(old_centroids)
        st.write("Proses clustering selesai.")
        st.pyplot(plot_clusters(data_standarized, labels, centroids, iteration))

        st.subheader("Hasil cluster akhir ")
        st.write(centroids)

        st.write("Perbandingan jika menggunakan library kmeans")
        kmeans = KMeans(k)
        kmeans.fit(data_standarized)
        centroids = kmeans.cluster_centers_
        st.write(pd.DataFrame(centroids, columns=feature).T)