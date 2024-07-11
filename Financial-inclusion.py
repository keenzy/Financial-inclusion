import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
#from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
import io
#from streamlit_pandas_profiling import st_profile_report

# give a title to our app 
st.title('Financial Inclusion Dataset') 
# Export data
st.header('Exploration des données du dataset')
data=pd.read_csv('Financial_inclusion_dataset.csv')
st.dataframe(data)
st.write("Infos du dataset")

buffer = io.StringIO()
data.info(buf=buffer)
s = buffer.getvalue()

st.text(s)
st.warning("data.info()/data.describe()")
st.write(data.info())
st.write(data.describe())
#profile = ProfileReport(data)
#st.subheader('Rapport d’analyse exploratoire des données')
#st_profile_report(profile)
#st.subheader('Gestion données manquantes')
st.write('Nombre de données manquantes')
st.warning('data.isnull().sum()')
st.write(data.isnull().sum())
st.success("Le dataset ne contient pas de valeurs manquantes")
st.write('Gestion des valeurs dupliquées')
st.warning('data.duplicated().sum()')
st.write(data.duplicated().sum())
st.success("Le dataset ne contient pas de valeurs dupliquées")


#Visualisation valeurs abberantes avec plotly boite à moustaches
st.subheader('Visualisation des valeurs abberantes avec box plots')

from plotly.subplots import make_subplots
import plotly.express as px

fig1 = px.box(data, y='age_of_respondent')
fig2 = px.box(data, y='household_size')
fig3 = px.box(data, y='year')

fig = make_subplots(rows=1, cols=3, subplot_titles=['age_of_respondent','household_size','year'])



fig.add_trace(fig1.data[0], row=1, col=1)
fig.add_trace(fig2.data[0], row=1, col=2)
fig.add_trace(fig3.data[0], row=1, col=3)


st.plotly_chart(fig, use_container_width=True)
st.success("Le dataset ne contient pas de valeurs corrompues ou aberrantes")
st.subheader("Encodage des données catégorielles")
#Encodage des données catégrorielles
data['country'] = data['country'].astype('category').cat.codes
data['education_level'] = data['education_level'].astype('category').cat.codes
data['job_type'] = data['job_type'].astype('category').cat.codes
data['marital_status'] = data['marital_status'].astype('category').cat.codes
data['location_type'] = data['location_type'].astype('category').cat.codes
data['cellphone_access'] = data['cellphone_access'].astype('category').cat.codes
data['gender_of_respondent'] = data['gender_of_respondent'].astype('category').cat.codes
data['relationship_with_head'] = data['relationship_with_head'].astype('category').cat.codes
data['bank_account'] = data['bank_account'].astype('category').cat.codes

data.drop('uniqueid',axis=1,inplace=True)
data.head()

#Entrainez les données
st.subheader("Entrainer les données")
st.write("Apercu sur la corréaltion entre la donnée cible et les données caractéristiques")
st.warning("data.corr()['bank_account']")

#Evaluer la corrélation entre la donnée cible et les données caractéristiques
data.corr()['bank_account']
#Choix donnée cible et features
st.write("Choix donnée cible et features")
st.warning("data.drop(columns=['bank_account','location_type','household_size','marital_status','relationship_with_head'], axis=1)")
st.warning("y = data['bank_account']")
X = data.drop(columns=['bank_account','location_type','household_size','marital_status','relationship_with_head'], axis=1)
y = data['bank_account']
# We split the data into training and test sets
st.write("Division des données training and test sets")
st.warning("X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#Testez un classifieur
st.subheader("Testez un classifieur de machine learning")

st.subheader("Create a kNN classifier")
st.warning("k = 5  # Choose the number of neighbors\n knn_classifier = KNeighborsClassifier(n_neighbors=k) \n ")
st.warning("k = 5  # Choose the number of neighbors")
k = 5  # Choose the number of neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Fit the model to the training data
st.warning("knn_classifier.fit(X_train, y_train")
knn_classifier.fit(X_train, y_train)

# Make predictions on the test data
st.warning("y_pred = knn_classifier.predict(X_test)")
y_pred = knn_classifier.predict(X_test)

# Evaluate accuracy
st.warning("accuracy = accuracy_score(y_test, y_pred)")
accuracy = accuracy_score(y_test, y_pred)
st.success(f"Accuracy: {accuracy:.2f}")
