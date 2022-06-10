import streamlit as st
import numpy as np

from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title('Aplikasi Mining TELKOM')
st.write("""
# menggunakan beberapa algoritma dan dataset yang berbeda
mana yang terbai??
""")

nama_dataset = st.sidebar.selectbox(
    'Pilih Dataset',
    ('Bunga IRIS', 'Kanker Payudara', 'Digit Angka')
)

st.write(f"## Dataset {nama_dataset}")

algoritma = st.sidebar.selectbox(
    'Pilih Algoritma',
    ('KNN', 'SVM', 'Random Forest')
)

def pilih_dataset(nama):
    data = None
    if nama == 'Bunga IRIS':
        data = datasets.load_iris()
    elif nama == 'Kanker Payudara':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_digits()
    x = data.data
    y = data.target
    return x,y

x, y = pilih_dataset(nama_dataset)
st.write('Jumlah Baris dan Kolom : ',x.shape)
st.write('Jumlah Kelas : ', len(np.unique(y)))

def tambah_parameter(nama_algoritma):
    params = dict()
    if nama_algoritma == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif nama_algoritma == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = tambah_parameter(algoritma)

def pilih_klasifikasi(nama_algoritma, params):
    algo = None
    if nama_algoritma == 'KNN':
        algo = KNeighborsClassifier(n_neighbors=params['K'])
    elif nama_algoritma == 'SVM':
        algo = SVC(C=params['C'])
    else:
        algo = RandomForestClassifier(n_estimators=params['n_estimators'],
        max_depth=params['max_depth'], random_state=1234)
    return algo
    
algo = pilih_klasifikasi(algoritma, params)

### PROSES KLASIFIKASI ###
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

algo.fit(x_train, y_train)
y_pred = algo.predict(x_test)

acc = accuracy_score(y_test, y_pred)


st.write(f'Algoritma = {algoritma}')
st.write(f'Akurasi = ', acc)


### PLOT DATASET ###
# pemproyeksikan data kedalam 2 komponen PCA
pca = PCA(2)
x_projected = pca.fit_transform(x)

x1 = x_projected[:, 0]
x2 = x_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')

plt.xlabel('Principal Component 1')
plt.xlabel('Principal Component 2')
plt.colorbar()

st.pyplot(fig)

### UPLOAD FILE ###
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
     # To read file as bytes:
     bytes_data = uploaded_file.getvalue()
     st.write(bytes_data)

     # To convert to a string based IO:
     stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
     st.write(stringio)

     # To read file as string:
     string_data = stringio.read()
     st.write(string_data)

     # Can be used wherever a "file-like" object is accepted:
     dataframe = pd.read_csv(uploaded_file)
     st.write(dataframe)

uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
for uploaded_file in uploaded_files:
     bytes_data = uploaded_file.read()
     st.write("filename:", uploaded_file.name)
     st.write(bytes_data)

