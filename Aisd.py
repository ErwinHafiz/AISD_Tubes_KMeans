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
    # tes sini datastandarize from dataselect
    centroid = data_standarized.apply(lambda x: float(x.sample()))
    st.write("random centroid untuk klaster ke -",i,":",centroid)
    centroids.append(centroid)
  st.write("Hasil centroid untuk masing - masing klaster :")
  return pd.concat(centroids, axis = 1 )


# def get labels distancte fungsi penentuan jarak dan pemberian label 
def get_labels (data,centroids):
  distances = centroids.apply(lambda x: np.sqrt(((data - x) ** 2 ).sum(axis = 1)))
  st.write("**Jarak Item & Label Klaster**")
  st.write(distances)
  st.write("Penentuan Label Kluster :")
  return distances.idxmin(axis =1 )

# def new centroid 
def new_centroids(data, labels, k):
  return   data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T




# display data 
def plot_clusters ( data, labels, centroids, iteration):
  pca = PCA(n_components = 2)
  data_2d = pca.fit_transform(data)
  centroids_2d = pca.transform(centroids.T)
  clear_output(wait = True)
  plt.title(f'Iteration {iteration}')
  plt.scatter(x=data_2d[:,0], y= data_2d[:,1], c=labels)
  plt.scatter(x=centroids_2d[:,0], y= centroids_2d[:,1])
  plt.show()

st.title("Kelompok 0 AISD-7:")
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
  st.write("Farid Xiaopang")
  st.write("Muhammad Pradipta Waskitha")
  st.write("Fikkry Ihza Fachrezi")



st.title("Penerapan algoritma K-Means pada dataset diabetes.xlsx")

tab1, tab2 = st.tabs(["Static data Kmeans", "Input data"])



with tab1:
  st.subheader("Data Diabetes")
  st.write(data)

  st.subheader("Feature yang digunakan untuk clustering")
  st.write(feature)

  st.write(data_select)
  st.write("Gambaran awal dataset diabetes")
  st.write(data_select.describe())



  st.subheader("Tahapan yang akan dilakukan dalam penerapan K-Means")



  # st.write("Tahapan: 4.update centroid")
  # st.write("Tahapan: 5.repeat 3-4 until stop")
  # st.write("Tahapan 2: Initialize random centroid")

# Tahapan input data, menentukan centroid, menentukan jarak, mengupdate centroid .repeat
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
  st.write(centroids)

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

  while iteration < max_iteration and not centroids.equals(old_centroids):
    st.write(f"Iterasi: {iteration}")
    old_centroids = centroids

    labels= get_labels(data_standarized, centroids)
    centroids = new_centroids(data_standarized, labels , k )

    plot_clusters(data_standarized, labels, centroids, iteration)
    iteration += 1


  st.write("Proses clustering selesai.")
  st.pyplot(plot_clusters(data_standarized, labels, centroids, iteration))

  st.subheader("hasil centroid akhir ")
  st.write(centroids)
    
  st.write("Perbandingan jika menggunakan library kmeans")
  kmeans = KMeans(3)
  kmeans.fit(data_standarized)
  centroids = kmeans.cluster_centers_

  # st.write(centroids)
  st.write(pd.DataFrame(centroids,columns=feature).T)
  

with tab2 :
  st.title("Ehe")