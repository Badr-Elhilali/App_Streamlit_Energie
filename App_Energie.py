import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


df=pd.read_csv("data/data_fi_region.csv")
prod_conso_jours=pd.read_csv('data/prod_conso_jours.csv')
data_meteo_prod_conso=pd.read_csv('data/data_fi.csv')
energie_clean_year=pd.read_csv('data/energie_clean_year.csv', )
prod_conso_jours['date'] = pd.to_datetime(prod_conso_jours['date'], errors='coerce')

# Configuration de la page Streamlit
st.set_page_config(page_title="Analyse de la consommation d'électricité", layout="wide")



# Menu de navigation
st.sidebar.title("Sommaire")

# Navigation
pages = ["Contexte du projet", "Exploration des données", "Analyse de données", "Modélisation","Prédiction"]
page = st.sidebar.radio("Aller vers la page :", pages)

# Ajouter de l'espace avec des lignes vides
st.sidebar.write("")
st.sidebar.write("")

# Ajouter un trait de séparation
st.sidebar.markdown("---")

# Ajouter encore un peu d'espace
st.sidebar.write("")

# Image et crédit
st.sidebar.image('data/image1.jpg', caption="Analyse énergétique basée sur les données de l'ODRE")
st.sidebar.write('Auteur : Badr El Hilali')

if page == pages[0]:
    # Contexte du projet
  st.title('Consommation en électricité de 2013-2022 en France')
  st.subheader('Présentation du sujet')

  st.write("""
        L’enjeu de la sécurité de l’approvisionnement en énergie est une question capitale. En effet, l’électricité a cette particularité de ne pas pouvoir être stockée en grande quantité.
        L'énergie électrique est aussi devenue un élément essentiel de la vie quotidienne et de l'activité économique moderne. Elle est utilisée pour alimenter les foyers, les industries et les transports.
        Pour répondre à la demande croissante d'électricité, les fournisseurs d'énergie ont besoin de prévoir avec précision la consommation électrique future. La prévision de la consommation d'électricité est un enjeu clé pour garantir la fiabilité du système électrique, optimiser la production et minimiser les coûts.
        
        La prévision de la consommation d'électricité repose sur l'analyse de données historiques, qui permettent de détecter des tendances et des cycles saisonniers dans la consommation électrique. Les variables clés pour la prévision de la consommation d'électricité sont la date et l'heure, qui permettent de tenir compte des fluctuations de la demande au cours du temps, ainsi que les données de production d'énergie, comme la production thermique, nucléaire et éolienne.
        
        Il est important de retenir que la quantité d’électricité produite et injectée dans le réseau doit toujours être en équilibre. À tout moment, nous devons avoir une consommation égale à la production. Tout déséquilibre provoquerait un blackout total.
        
    """)

    # Objectifs du projet
  expander = st.expander("Objectifs")
  expander.markdown("""
        * Comprendre la relation entre la consommation et la production des différentes sources d'énergie électrique
        * Focus sur les énergies renouvelables
        * Produire un modèle permettant de calculer les estimations des consommations 
    """)

    # Affichage d'une image dans l'application Streamlit
elif page == pages[1]:
  st.title('Exploration des données ')
  st.sidebar.markdown("---")
    
  st.subheader("Identification des sources de données")
  st.write("""
        Le jeu de données dont nous disposons concerne la consommation d'énergie électrique en France ainsi que la production des différentes 
        sources d'énergie électrique, telles que l'énergie nucléaire, thermique et éolienne. Ces données sont collectées à une résolution horaire 
        sur une période de plusieurs années. Les données principales proviennent du portail d’Open Data Réseaux Énergies.

        Les données de consommation présentent les données régionales depuis l'année 2013.  
        Nous y trouverons les données de consommation et la production selon les différentes filières composant le mix énergétique avec des mesures 
        prises toutes les demi-heures.
""")

# Ajout de données complémentaires
  st.subheader("Ajout de données complémentaires")
  st.markdown("""
    Afin d'enrichir l'analyse de la consommation énergétique, des données météorologiques ont été ajoutées, telles que :
    * Température
    * Vitesse du vent
    * Et plusieurs autres variables météorologiques
""")
 
  # Exploration des données
  st.subheader("Présentation finale des données")
    
  st.dataframe(df.head())
  st.write("Dimensions du dataframe :")
    
  st.write(df.shape)
    
  if st.checkbox("Afficher les valeurs manquantes") : 
    st.dataframe(df.isna().sum())
        
  if st.checkbox("Afficher les doublons") : 
    st.write(df.duplicated().sum())

elif page == pages[2]:
  st.title('Analyse de données')
  st.sidebar.markdown("---")

  


  # Titre de la section
  st.header("La matrice de corrélation des variables")

  # Sélection des colonnes numériques
  colonnes_numeriquess = [
   'production_totale',
   'Température (°C)',
   'Humidité', 
   'Vitesse du vent moyen 10 mn',
   'Direction du vent moyen 10 mn',
   'Pression station',
   'Consommation (MW)',
   'Thermique (MW)',
   'Nucléaire (MW)',
   'Eolien (MW)', 
   'Solaire (MW)',
   'Hydraulique (MW)',
   'Pompage (MW)',
   'Bioénergies (MW)',
   'Ech. physiques (MW)'
]

  # Calcul de la matrice de corrélation
  correlation_matrix = data_meteo_prod_conso[colonnes_numeriquess].corr()

  # Création de la figure
  fig, ax = plt.subplots(figsize=(15, 6))

# Création de la heatmap
  sns.heatmap(correlation_matrix,
           annot=True,  # Afficher les valeurs
           cmap='coolwarm',  # Palette de couleurs 
           center=0,  # Centre la palette sur 0
           fmt='.2f',  # Format des nombres (2 décimales)
           linewidths=0.5,  # Largeur des lignes
           ax=ax
           )

  # Personnalisation
  plt.title('Matrice de Corrélation - Variables Météorologiques et Production d\'Énergie (National)')
  plt.xticks(rotation=45, ha='right')
  plt.yticks(rotation=0)
  plt.tight_layout()

  # Affichage dans Streamlit
  st.pyplot(fig)

 
  # Titre de la section
  st.header("Comparaison Production vs Consommation")

# Création de la figure
  fig, ax = plt.subplots(figsize=(15, 6))

# Création du graphique de comparaison
  sns.lineplot(data=prod_conso_jours, x='date', y='production_totale', 
            label='Production Totale', ax=ax)
  sns.lineplot(data=prod_conso_jours, x='date', y='Consommation (MW)', 
            label='Consommation', ax=ax)

# Personnalisation
  plt.title('Comparaison Production Totale vs Consommation')
  plt.xticks(rotation=45) 
  plt.legend()
  plt.tight_layout()

# Affichage dans Streamlit
  st.pyplot(fig)

# Optionnel : Ajouter une description sous le graphique
  st.write("""
  On peut observer des périodes où la production dépasse la consommation (surplus) et d'autres où la consommation dépasse la production (déficit).
Cette comparaison est essentielle pour comprendre l'équilibre énergétique du pays. Les périodes de déficit peuvent nécessiter des importations d'électricité ou des mesures de réduction de la demande, tandis que les périodes de surplus peuvent être l'occasion d'exporter de l'électricité ou de stocker des réserves.
  """)


  annees = np.arange(2013, 2023, 1)

# Interface Streamlit
  st.title("📊 Visualisation de la Production & Consommation d'Énergie")

# Sélection de l'année avec un slider
  annee_selectionnee = st.slider("Sélectionnez une année :", int(annees.min()), int(annees.max()), int(annees.min()))

# ✅ Vérifier que la colonne "date" est bien en datetime
  if prod_conso_jours['date'].dtype == 'datetime64[ns]':
    mask = prod_conso_jours['date'].dt.year == annee_selectionnee
    data_annee = prod_conso_jours[mask]

    # Vérifier si des données existent pour l'année sélectionnée
    if not data_annee.empty:
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(data_annee['date'], data_annee['production_totale'], label='Production', color='blue')
        ax.plot(data_annee['date'], data_annee['Consommation (MW)'], label='Consommation', color='orange')

        ax.set_title(f'Production & Consommation en {annee_selectionnee}')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)
        st.write("""Grâce à ce graphique nous confirmons un effet de saisonnalité chaque année !. Cela nous permet également de remarquer qu'il existe un effet de saisonnalité  par mois. On conclus que la production et la consommation semble se suivre de manière cyclique.""")
    else:
        st.warning(f"Aucune donnée disponible pour l'année {annee_selectionnee}.")
  else:
    st.error("Erreur : La colonne 'date' n'est pas au format datetime.")







  dfgeo = df.groupby('Région', as_index=False).agg({
    'production_totale': 'mean',
    'Thermique (MW)': 'mean',
    'Nucléaire (MW)': 'mean', 
    'Eolien (MW)': 'mean',
    'Solaire (MW)': 'mean',
    'Hydraulique (MW)': 'mean'
})

# Interface Streamlit
  st.title("🌍 Comparaison de la Production Énergétique par Région")

# Création du graphique Plotly
  fig = go.Figure()

# Ajout de la production totale
  fig.add_trace(go.Bar(
    name='Production Totale',
    x=dfgeo['Région'],
    y=dfgeo['production_totale'],
    marker_color='rgba(70, 130, 180, 0.7)',
    width=0.5
))

# Ajout des différents types d'énergie
  types_energie = ['Thermique (MW)', 'Nucléaire (MW)', 'Eolien (MW)', 
                'Solaire (MW)', 'Hydraulique (MW)']
  colors = ['rgb(255,127,14)', 'rgb(44,160,44)', 'rgb(214,39,40)', 
          'rgb(148,103,189)', 'rgb(140,86,75)']

  for type_energie, color in zip(types_energie, colors):
    fig.add_trace(go.Bar(
      name=type_energie.replace(' (MW)', ''),
      x=dfgeo['Région'],
      y=dfgeo[type_energie],
      marker_color=color,
      opacity=0.7
    ))

# Mise en page
  fig.update_layout(
    xaxis_tickangle=45,
    barmode='group',
    height=600,
    width=1200,
    template='plotly_white',
    showlegend=True,
    legend=dict(
       yanchor="top",
       y=0.99,
      xanchor="right",
      x=0.99
    ),
    yaxis_title="Production (MW)"
)

# Affichage du graphique dans Streamlit
  st.plotly_chart(fig, use_container_width=True)
  
  st.write("- **Production d'énergie variable :** Les régions françaises présentent des niveaux de production d'énergie très différents. Certaines régions, comme Auvergne-Rhône-Alpes, affichent une production élevée, tandis que d'autres, comme l'Île-de-France, ont une production très faible.")
  st.write("- **Mix énergétique diversifié :** Chaque région a une combinaison unique de sources d'énergie. On observe une prédominance du nucléaire et une contribution notable des énergies renouvelables (éolien, solaire) dans plusieurs régions.")
  st.write("- **Dépendance énergétique :** Les régions avec une faible production, comme l'Île-de-France, dépendent fortement des autres régions pour leur approvisionnement énergétique.")

  st.subheader("La production en MW d'électricité")
  option=st.radio(
        "Set selectbox label visibility 👉",
        key="visibility",
        options=['Thermique (MW)', 
          'Nucléaire (MW)', 
          'Eolien (MW)',
            'Solaire (MW)', 
            'Hydraulique (MW)', 
            'Pompage (MW)',
            'Bioénergies (MW)'],
    )
 #bloquer les axes 
  def lineplot():

    if option =='Thermique (MW)':
        st.bar_chart(data=energie_clean_year[['Thermique (MW)','annee']],x='annee')
        st.write("Production thermique : La production d'électricité à partir de sources thermiques semble avoir été la principale source d'énergie au début de la période étudiée, mais on peut observer une diminution progressive de cette production au fil des années.")
    elif option=='Nucléaire (MW)':
        st.bar_chart(data=energie_clean_year[['Nucléaire (MW)','annee']],x='annee')
        st.write("Production nucléaire : La production d'électricité à partir de sources nucléaires a maintenu une certaine stabilité au cours des années, avec des variations mineures")
    elif option=='Eolien (MW)':
        st.bar_chart(data=energie_clean_year[['Eolien (MW)','annee']],x='annee')
    elif option=='Solaire (MW)':
        st.bar_chart(data=energie_clean_year[['Solaire (MW)','annee']],x='annee')
        st.write("Énergie éolienne et solaire : Les sources d'énergie éolienne et solaire ont connu une augmentation significative de leur production d'électricité au fil du temps, reflétant une adoption croissante des énergies renouvelables")
    elif option=='Hydraulique (MW)':
        st.bar_chart(data=energie_clean_year[['Hydraulique (MW)','annee']],x='annee')
    elif option=='Pompage (MW)':
        st.bar_chart(data=energie_clean_year[['Pompage (MW)','annee']],x='annee')
        st.write("Production hydraulique et pompage : La production d'électricité à partir de sources hydrauliques et de pompage a montré des variations en fonction des conditions climatiques et des besoins en électricité.")
    elif option=='Bioénergies (MW)':
        st.bar_chart(data=energie_clean_year[['Bioénergies (MW)','annee']],x='annee')
        st.write("Bioénergies : La production d'électricité à partir de sources de bioénergies a également connu une augmentation constante, soulignant l'intérêt croissant pour les sources d'énergie renouvelables.")
  lineplot()


  st.title("📊 Évolution de la consommation d'électricité par région")

# Assurez-vous que la colonne date est au format datetime
  df_grouped = df.groupby(['date', 'Région'])['Consommation (MW)'].sum().reset_index()
  if not pd.api.types.is_datetime64_any_dtype(df_grouped['date']):
      df_grouped['date'] = pd.to_datetime(df_grouped['date'])

  fig, ax = plt.subplots(figsize=(15, 8))
      
  # Get unique regions
  regions = df_grouped['Région'].unique()
      
  # Plot each region
  for region in regions:
      df_region = df_grouped[df_grouped['Région'] == region]
      ax.plot(df_region['date'], df_region['Consommation (MW)'], label=region)

  # Customize the plot
  ax.set_title("📈 Évolution de la consommation d'électricité par région")
  ax.set_xlabel("Date")
  ax.set_ylabel("Consommation (MW)")
  ax.legend(title="Région", bbox_to_anchor=(1.05, 1), loc='upper left')
  ax.grid(True)

  # Simplifier la gestion des dates - utiliser le formateur le plus simple
  import matplotlib.dates as mdates
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

  # Réduire le nombre de ticks à environ 10 (une par année pour la période 2013-2022)
  ax.xaxis.set_major_locator(plt.MaxNLocator(10))

  # Désactiver la rotation pour meilleure lisibilité
  plt.xticks(rotation=0)

  plt.tight_layout()
  st.pyplot(fig)








  consommation_par_region = df.groupby('Région')['Consommation (MW)'].mean()

# Labels des régions
  regions = ['Auvergne-Rhône-Alpes', 'Bourgogne-Franche-Comté', 'Bretagne', 
           'Centre-Val de Loire', 'Grand Est', 'Hauts-de-France', 'Normandie',
           'Nouvelle-Aquitaine', 'Occitanie', 'Pays de la Loire',
           'Provence-Alpes-Côte d Azur', 'Île-de-France']

# Couleurs personnalisées
  colors = ["#95B2F8", "#28666e", "#7c9885", "#b5b682", "#8b3d63", "#a682ff",
          "#5887ff", "#55c1ff", "#f3c178", "#fe5e41", 'pink', '#366fa2']

# Création de la figure Matplotlib
  fig, ax = plt.subplots(figsize=(6, 6))  # Augmenter la taille de la figure pour plus d'espace
  wedges, texts, autotexts = ax.pie(
    x=consommation_par_region, 
    labels=regions,
    colors=colors,
    autopct=lambda x: str(round(x, 2)) + '%',
    pctdistance=0.75,  # Ajuster la distance des pourcentages
    labeldistance=1.05,  # Ajuster la distance des étiquettes
    shadow=True,
)



#

  fig.patch.set_facecolor('white')  # Définir le fond en blanc

# Interface Streamlit
  st.title("📊 Consommation d'Énergie par Région")
  st.pyplot(fig)






  # Titre principal
  st.title("Analyse de la consommation électrique")
  st.write("Ce graphique explore la relation entre la consommation et deux facteurs environnementaux : la température et l'humidité.")
  df['date'] = pd.to_datetime(df['date'], errors='coerce')
# Création de la colonne mois (à adapter selon votre dataframe)
  df['mois'] = df['date'].dt.month

# Création de deux colonnes pour afficher les graphiques côte à côte
  col1, col2 = st.columns(2)

  with col1:
    fig1 = px.scatter(
      df, x='Température (°C)', y='Consommation (MW)', color='mois',
      title="🌡️ Consommation en fonction de la température",
      labels={'Température (°C)': 'Température (°C)', 'Consommation (MW)': 'Consommation (MW)', 'mois': 'Mois'},
      color_continuous_scale="turbo"  # 🎨 Changer la palette de couleurs
    )
    fig1.update_layout(width=600, height=500, showlegend=True)
    st.plotly_chart(fig1)

  with col2:
    fig2 = px.scatter(
        df, x='Humidité', y='Consommation (MW)', color='mois',
        title="💧 Consommation en fonction de l'humidité",
        labels={'Humidité': 'Humidité (%)', 'Consommation (MW)': 'Consommation (MW)', 'mois': 'Mois'},
        color_continuous_scale="plasma"  # 🎨 Palette différente pour un meilleur contraste
    )
    fig2.update_layout(width=600, height=500, showlegend=True)
    st.plotly_chart(fig2)
  st.write("La température semble avoir une influence plus marquée sur la consommation que l'humidité. Les variations saisonnières (chauffage en hiver, climatisation en été) jouent un rôle important dans la relation entre ces facteurs et la consommation.")





    
   
elif page == pages[3]: 
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.model_selection import train_test_split,cross_val_score, learning_curve
  from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
  import xgboost as xgb 
  import scipy.stats as stats

  from sklearn.linear_model import LinearRegression

  import joblib
  df=pd.read_csv("data/data_fi_region.csv")

# --- 📌 Charger les données ---
  df.drop('Unnamed: 0', axis=1, inplace=True)
  
  df['date'] = pd.to_datetime(df['date'])
  df['jour_semaine'] = df['date'].dt.dayofweek
  df['mois'] = df['date'].dt.month
  df.set_index('date',inplace=True)
  df = df[['Température (°C)', 'Humidité', 'Vitesse du vent moyen 10 mn', 'Pression station', 'Région', 'Consommation (MW)','jour_semaine', 'mois']]
  regions = {
    'Île-de-France': 75,
    'Auvergne-Rhône-Alpes': 69,
    'Bourgogne-Franche-Comté': 21,
    'Bretagne': 29,
    'Centre-Val de Loire': 37,
    'Grand Est': 67,
    'Hauts-de-France': 59,
    'Normandie': 76,
    'Nouvelle-Aquitaine': 33,
    'Occitanie': 31,
    'Pays de la Loire': 44,
    "Provence-Alpes-Côte d'Azur": 13
}
  df['Région'] = df['Région'].map(regions)
  X = df.drop(columns=['Consommation (MW)'])
  y= df['Consommation (MW)']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  rf_model = joblib.load("rfo_model")
  xgb_model = joblib.load("xgbo_model")
  lr_model = joblib.load("lr_model")

  y_pred_rf = rf_model.predict(X_test)
  y_pred_xgb = xgb_model.predict(X_test)
  y_pred_lr = lr_model.predict(X_test)
  metrics = {
    "Modèle": ["RandomForestRegressor", "XGBoostRegressor","LinearRegression"],
    "R² Score Test": [r2_score(y_test, y_pred_rf), r2_score(y_test, y_pred_xgb),r2_score(y_test,y_pred_lr)],
    "RMSE": [np.sqrt(mean_squared_error(y_test, y_pred_rf)), np.sqrt(mean_squared_error(y_test, y_pred_xgb)),np.sqrt(mean_squared_error(y_test, y_pred_lr))],
    "MAE": [mean_absolute_error(y_test, y_pred_rf), mean_absolute_error(y_test, y_pred_xgb),mean_absolute_error(y_test, y_pred_lr)],
    "R² Score Train": [rf_model.score(X_train, y_train), xgb_model.score(X_train, y_train),lr_model.score(X_train, y_train)]
}

  df_metrics = pd.DataFrame(metrics)
  st.title("📊 Comparaison des Modèles de Prédiction")

  st.write("### 🔍 Objectif : Prédire la consommation d'électricité par région")
  st.write("Nous comparons **Random Forest** et **XGBoost** pour voir lequel est le plus performant.")

  # Affichage du tableau
  st.write("### 📋 Tableau comparatif des modèles")
  st.table(df_metrics)

  st.write("""
🏆 **Analyse des résultats :**  
Les modéles Random Forest , XGboost et ont les meilleures performances. Mais le score trop élevé de Random Forest ressemble à du suraprentissage. Nous séllectionons donc le  **'XGboost'** pour son score un peu plus faible que  Random Forest. 
""")

 

  feature_importance_xgb = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=True)

# Convertir en pourcentage
  feature_importance_xgb['Importance (%)'] = feature_importance_xgb['Importance'] * 100

  # --- 📊 Graphique interactif avec Plotly ---
  st.title("📊 Importance des Caractéristiques ")

  fig = px.bar(
      feature_importance_xgb, 
      x="Importance (%)", 
      y="Feature", 
      orientation="h", 
      title="🔴 Feature Importance",
      color="Importance (%)",
      color_continuous_scale="blues",
      hover_data={"Importance (%)": ":.2f"}  # Afficher uniquement au survol
  )

  # Supprimer les labels de texte sur les barres
  fig.update_traces(textposition="none")  

  fig.update_layout(
      xaxis_title="Importance (%)", 
      yaxis_title="Caractéristiques", 
      hovermode="x unified"  # Pour un affichage clair au survol
  )

  # --- 📌 Affichage Streamlit ---
  st.plotly_chart(fig)


  cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='r2')


  # --- 📌 Calcul de la courbe d'apprentissage ---
  train_sizes, train_scores, val_scores = learning_curve(
      estimator=xgb_model,
      X=X,
      y=y,
      train_sizes=np.linspace(0.1, 1.0, 10),  # Taille des ensembles d'entraînement
      cv=5,  # Validation croisée avec 5 folds
      scoring='r2',  # Métrique utilisée (R²)
      n_jobs=-1  # Utilisation de tous les cœurs disponibles
  )

  # Calcul des moyennes et écarts-types
  train_scores_mean = train_scores.mean(axis=1)
  train_scores_std = train_scores.std(axis=1)
  val_scores_mean = val_scores.mean(axis=1)
  val_scores_std = val_scores.std(axis=1)

  # --- 📊 Graphique interactif avec Plotly ---
  fig = go.Figure()

  # Ajout des courbes
  fig.add_trace(go.Scatter(
      x=train_sizes, y=train_scores_mean, mode='lines+markers', name="Score d'entraînement",
      line=dict(color='blue')
  ))
  fig.add_trace(go.Scatter(
      x=train_sizes, y=val_scores_mean, mode='lines+markers', name="Score de validation",
      line=dict(color='orange')
  ))

  # Ajout des bandes d'erreur
  fig.add_trace(go.Scatter(
      x=train_sizes, y=train_scores_mean + train_scores_std, mode='lines', fill=None, line=dict(color='blue', dash='dot'), showlegend=False
  ))
  fig.add_trace(go.Scatter(
      x=train_sizes, y=train_scores_mean - train_scores_std, mode='lines', fill='tonexty', line=dict(color='blue', dash='dot'), showlegend=False
  ))

  fig.add_trace(go.Scatter(
      x=train_sizes, y=val_scores_mean + val_scores_std, mode='lines', fill=None, line=dict(color='orange', dash='dot'), showlegend=False
  ))
  fig.add_trace(go.Scatter(
      x=train_sizes, y=val_scores_mean - val_scores_std, mode='lines', fill='tonexty', line=dict(color='orange', dash='dot'), showlegend=False
  ))

  # Mise en forme du graphique
  fig.update_layout(
      title="📈 Courbe d'Apprentissage",
      xaxis_title="Taille de l'ensemble d'entraînement",
      yaxis_title="Score R²",
      hovermode="x unified",
      template="plotly_white"
  )

  # --- 📌 Affichage Streamlit ---
  st.plotly_chart(fig)



  assert len(y_pred_xgb) == len(y_test), "Les tailles de y_pred_xgb et y_test ne correspondent pas !"

  # --- 📌 Création du DataFrame de comparaison ---
  comparaison_df = pd.DataFrame({
      'Réelle': y_test.values,
      'Prédiction': y_pred_xgb,
      'Jour de la semaine': X_test['jour_semaine'],
      'Mois': X_test['mois']
  }, index=X_test.index)

  # --- 📌 Calcul des moyennes ---
  moyenne_par_jour = comparaison_df.groupby('Jour de la semaine')[['Réelle', 'Prédiction']].mean()
  moyenne_par_mois = comparaison_df.groupby('Mois')[['Réelle', 'Prédiction']].mean()
  st.write("📊 **Consommation Réelle vs Prédiction par différentes dimensions**")

  # --- 📌 Sélecteur pour le filtre ---
  choix = st.radio("Sélectionner un filtre :", ["Jour de la semaine", "Mois"])

  # --- 📌 Affichage des résultats ---
  if choix == "Jour de la semaine":
      fig = px.bar(moyenne_par_jour, x=moyenne_par_jour.index, y=['Réelle', 'Prédiction'],
                  title="Moyenne de la consommation par jour de la semaine",  
                  labels={'value': "Consommation (MW)", 'variable': "Type"},
                  barmode='group', 
                  color_discrete_map={'Réelle': '#87ceeb', 'Prédiction': '#002855'},  # Couleurs modifiées
                  width=1300,  # Largeur du graphique
                  height=700)  # Hauteur du graphique
      st.plotly_chart(fig)

  else:
      fig = px.bar(moyenne_par_mois, x=moyenne_par_mois.index, y=['Réelle', 'Prédiction'],
                  title="Comparaison Moyenne - Mois",
                  labels={'value': "Consommation (MW)", 'variable': "Type"},
                  barmode='group', 
                  color_discrete_map={'Réelle': '#87ceeb', 'Prédiction': '#002855'},  # Couleurs modifiées
                  width=1300,  # Largeur du graphique
                  height=700)  # Hauteur du graphique
      st.plotly_chart(fig)



  st.title("📊 Etude des residus ")
  residus = y_test - y_pred_xgb

# --- 🎨 Création des graphiques ---
  st.write("### Visualisation des résultats")

  # Diviser la page en 2 colonnes pour les graphiques
  col1, col2 = st.columns(2)

  # Graphique 1 : Comparaison des valeurs réelles vs prédites
  with col1:
      fig1, ax1 = plt.subplots()
      ax1.scatter(y_test, y_pred_xgb, alpha=0.5)
      ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
      ax1.set_xlabel('Valeurs Réelles')
      ax1.set_ylabel('Valeurs Prédites')
      ax1.set_title('Comparaison des valeurs réelles vs prédites')
      st.pyplot(fig1)

  # Graphique 2 : Graphique de dispersion des résidus
  with col2:
      fig2, ax2 = plt.subplots()
      ax2.scatter(y_pred_xgb, residus, alpha=0.5)
      ax2.axhline(y=0, color='r', linestyle='--')
      ax2.set_xlabel('Valeurs Prédites')
      ax2.set_ylabel('Résidus')
      ax2.set_title('Dispersion des Résidus')
      st.pyplot(fig2)

  # Diviser la page en 2 colonnes pour les graphiques suivants
  col3, col4 = st.columns(2)

  # Graphique 3 : Histogramme des résidus
  with col3:
      fig3, ax3 = plt.subplots()
      sns.histplot(residus, kde=True, ax=ax3)
      ax3.set_xlabel('Résidus')
      ax3.set_ylabel('Fréquence')
      ax3.set_title('Histogramme des résidus')
      st.pyplot(fig3)

  # Graphique 4 : QQ plot des résidus
  with col4:
      fig4, ax4 = plt.subplots()
      stats.probplot(residus, dist="norm", plot=ax4)
      ax4.set_title('QQ Plot des Résidus')
      st.pyplot(fig4)

elif page == pages[4]:
  from datetime import datetime
  import joblib
  regions = {
    'Île-de-France': 75, 'Auvergne-Rhône-Alpes': 69, 'Bourgogne-Franche-Comté': 21,
    'Bretagne': 29, 'Centre-Val de Loire': 37, 'Grand Est': 67, 'Hauts-de-France': 59,
    'Normandie': 76, 'Nouvelle-Aquitaine': 33, 'Occitanie': 31, 'Pays de la Loire': 44,
    "Provence-Alpes-Côte d'Azur": 13
}
  st.title('Projet Energie - Prédictions')

# Sélection de la région
  region_selectionnee = st.selectbox('Sélectionnez une région :', list(regions.keys()))

# Ajustement de la température
  temperature = st.slider('Ajustez la température (°C) :', -10, 30, 15)

  # Sélection de la date (sans heure)
  date_selectionnee = st.date_input('Sélectionnez une date :', datetime(2022, 1, 1))

  # Affichage des informations sélectionnées
  st.write(f"Date sélectionnée : {date_selectionnee}")
  st.write(f"**Le {date_selectionnee} en {region_selectionnee} :**")

  # Préparation des données pour la prédiction
  donnees_utilisateur = {
      'Température (°C)': temperature,
      'Humidité': df['Humidité'].mean(),  # Vous pouvez ajuster cela selon vos besoins
      'Vitesse du vent moyen 10 mn': df['Vitesse du vent moyen 10 mn'].mean(),  # Idem
      'Pression station': df['Pression station'].mean(),  # Idem
      'Région': regions[region_selectionnee],
      'jour_semaine': pd.to_datetime(date_selectionnee).dayofweek,  # Jour de la semaine (0 = Lundi, 6 = Dimanche)
      'mois': pd.to_datetime(date_selectionnee).month  # Mois (1 = Janvier, 12 = Décembre)
  }
  xgb_model = joblib.load("xgbo_model")


  # Conversion en DataFrame
  donnees_utilisateur_df = pd.DataFrame([donnees_utilisateur])

  # Prédiction
  prediction = xgb_model.predict(donnees_utilisateur_df)

  # Affichage de la prédiction
# Affichage de la prédiction en gras et avec un style original
  st.markdown(
      f"<h2 style='text-align: center; color: #FF5733; font-weight: bold;'>"
      f"Prédiction : {prediction[0]:.1f} MW"
      f"</h2>",
      unsafe_allow_html=True
  )
