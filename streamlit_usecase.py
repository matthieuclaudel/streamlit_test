import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df=pd.read_csv("/Users/matthieuclaudel/Documents/Formation_MLE/sprint/sprint10/Streamlit/train.csv")

st.title("Projet de classification binaire Titanic")
st.sidebar.title("Sommaire")

# création des différentes pages
pages=["Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)


# CONSTRUCTION PAGE 0 - EXPLORATION
# titre de la page Exploration
if page == pages[0] : 
    st.write("### Introduction")
    st.write("##### Tableau")
    st.dataframe(df.head(10))
    
    st.write("##### Données sur le jeu")
    st.write(df.shape)
    st.dataframe(df.describe())
    st.dataframe(df.info())
    
    if st.checkbox("Afficher les NA") : # case à cocher pour effectuer une action
        st.dataframe(df.isna().sum())


# CONSTRUCTION PAGE 1 - DATAVIZ
if page == pages[1] : 
    st.write("### DataVizualization")
    
    fig = plt.figure()
    sns.countplot(x = 'Survived', data = df)
    st.pyplot(fig) # commande pour affichage figure
    
    fig = plt.figure()
    sns.countplot(x = 'Sex', data = df)
    plt.title("Répartition du genre des passagers")
    st.pyplot(fig)
    
    fig = plt.figure()
    sns.countplot(x = 'Pclass', data = df)
    plt.title("Répartition des classes des passagers")
    st.pyplot(fig)
    
    fig = sns.displot(x = 'Age', data = df)
    plt.title("Distribution de l'âge des passagers")
    st.pyplot(fig)
    
    fig = plt.figure()
    sns.countplot(x = 'Survived', hue='Sex', data = df)
    st.pyplot(fig)
    
    fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
    st.pyplot(fig)
    
    fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    df_corr = df[['Survived','Pclass','Age',]]
    sns.heatmap(df_corr.corr(), ax=ax)
    st.pyplot(fig) # affichage matrice de correlation /!\ on utilise st.write
    
# CONSTRUCTION PAGE MODELISATION
if page == pages[2] : 
    st.write("### Modelisation")
    
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    y = df['Survived'] # déf variable target
    X_cat = df[['Pclass', 'Sex',  'Embarked']] # def variable catégorielles
    X_num = df[['Age', 'Fare', 'SibSp', 'Parch']] # def variables numériques
    
    # 
    for col in X_cat.columns:
        X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0]) # remplacement des valeurs manquantes par le mode le plus fréquent
    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].median()) # remplacement des Nan par la médiane
    
    X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns) # Encodage des variables catégorielles en dummies (binaires)
    
    X = pd.concat([X_cat_scaled, X_num], axis = 1) # concaténation pour création X propre
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler() # Normalisation standard des donénes
    X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
    X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])
    
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    
    from sklearn.metrics import confusion_matrix
    
    def prediction(classifier): # définition d'une fonction portant les différent modèles utilisés
        if classifier == 'Random Forest':
            clf = RandomForestClassifier()
        elif classifier == 'SVC':
            clf = SVC()
        elif classifier == 'Logistic Regression':
            clf = LogisticRegression()
        clf.fit(X_train, y_train)
        return clf # retourne le modèle entraîné
    
    def scores(clf, choice): # définition d'une fonction qui permet le choix de la métrique
        if choice == 'Accuracy':
            return clf.score(X_test, y_test)
        elif choice == 'Confusion matrix':
            return confusion_matrix(y_test, clf.predict(X_test))
        
    choix = ['Random Forest', 'SVC', 'Logistic Regression']
    option = st.selectbox('Choix du modèle', choix) # création d'une boite de sélection
    st.write('Le modèle choisi est :', option)
    
    clf = prediction(option)
    display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
    if display == 'Accuracy':
        st.write(scores(clf, display))
    elif display == 'Confusion matrix':
        st.dataframe(scores(clf, display))