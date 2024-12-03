# UrbanAnalisisProject

# UrbanAnalisisProject

## Objectives
- Analyze the relationship between these variables and potentially predict or classify certain neighborhood characteristics.
- Understand the relationships between variables.
- Predict pollution.
- Identify groups of neighborhoods.

## Dataset

### Variables
- **Total area (km²)**
- **Vegetation area (km²)**
- **Fine particles pollution pm25 (mcg/m³)**

### Description
- **79 neighborhoods/roundabouts** in 3 different cities:
  - **NYC**: 42
  - **Berlin**: 12
  - **Seoul**: 25

The data was collected from various sources by myself, I had to process them and put them in a single CSV file (`data/data.csv`).

### Data Sources
- **Surface area**: [Wikipedia](https://www.wikipedia.org/)

- **Seoul**: [Seoul Open Data Plaza](https://data.seoul.go.kr/dataService/boardList.do)
- **New York City**: [NYC Health](https://a816-dohbesp.nyc.gov/IndicatorPublic/)
- **Berlin**:
  - [Oasis Hub](https://oasishub.co/dataset/berlin-germany-district-level-environmental-database/resource/be1739c5-1c58-4199-be5a-ea6f15299cb5?inner_span=True)
  - [Berlin Environmental Data](https://www.berlin.de/sen/uvk/_assets/natur-gruen/stadtgruen/daten-und-fakten/ausw_14.pdf)


<!-->

Notes: 
vérifications de 2 manières différentes
avec la librairie iforest(with 5%) et en faisant le calcul IQR same results
, sur plus de deux variables
j'ai decider de garder les outliers car cela ne correspond pas à une erreur mais à une valeur légitime (plus grande superficie comme Steglitz-Zehlendorf et Treptow-Kopenick), quand j'ai tester avec les pm/m3 les outliers n'étaient pas bien plus grand ou petits que les autres (différence <1)
j'ai aussi décider de les garder car je n'ai pas beaucoup de valeurs 

PCA
L'axe des abscisses (PC1) correspond à la première composante principale, qui explique 54,73 % de la variance totale des données.
L'axe des ordonnées (PC2) correspond à la deuxième composante principale, expliquant 33,08 % de la variance.
Ensemble, ces deux composantes capturent environ 87,81 % de la variance totale (54,73 % + 33,08 %), ce qui est suffisant pour une bonne représentation des données dans un espace réduit à deux dimensions.

Dispersion des points :

    NYC (bleu foncé) : Les points sont assez regroupés près du centre. Cela indique une relative homogénéité des observations pour cette ville.
    S (vert clair) : Les points sont dispersés sur la droite du graphe, formant un groupe bien distinct des autres villes.
    B (orange) : Les points se trouvent entre NYC et S, mais forment un nuage plus dispersé.

b. Séparation entre groupes :

    Les contours montrent des différences dans la répartition des observations pour chaque ville. Cela suggère que les données contiennent des caractéristiques discriminantes permettant de séparer les villes.
        S semble bien distinct de NYC et B, avec des points éloignés sur l'axe PC1.
        NYC et B présentent un certain chevauchement, mais les contours indiquent qu'ils ont des distributions légèrement différentes.
Clusters naturels :

    Le graphe suggère qu’il existe des différences significatives entre les villes (NYC, S, et B), probablement en raison des caractéristiques initiales (pollution, densité végétale, etc.).
    Les clusters peuvent être liés à des facteurs spécifiques qui distinguent ces villes, comme des différences géographiques, économiques ou environnementales.

b. Potentiel d’un modèle supervisé :

    Puisque les groupes semblent bien séparés (surtout S), cela indique qu’un modèle de classification supervisé (par ex. Random Forest ou SVM) pourrait bien distinguer les villes sur la base des caractéristiques initiales.

c. Analyse exploratoire :

    Le chevauchement entre NYC et B pourrait indiquer des similitudes dans les caractéristiques (par exemple, une pollution similaire ou une densité végétale comparable).
    Le groupe S distinct pourrait refléter des caractéristiques uniques propres à cette ville (par ex., moins polluée, plus de végétation).


 Explorer la séparation en clustering :

    Tester des algorithmes de clustering (K-means, DBSCAN) pour valider si les clusters correspondent bien aux villes.

c. Validation supervisée :

    Si les villes sont utilisées comme classes, entraîner un modèle supervisé et évaluer sa précision sur ces données.





Analyse Non Supervisée

    Clustering:
        Objectif: Regrouper les quartiers en fonction de leur similarité sur les variables.
        Algorithmes: K-means, DBSCAN, Hierarchical Clustering.
        Intérêt: Identifier des clusters de quartiers avec des caractéristiques environnementales similaires (par exemple, quartiers très verts avec une faible pollution, quartiers densément peuplés avec une forte pollution).
    Analyse en Composantes Principales (ACP):
        Objectif: Réduire la dimensionnalité des données et identifier les principales sources de variation.
        Intérêt: Visualiser les quartiers dans un espace à plus faible dimension pour mieux comprendre leurs relations.

Analyse Supervisée

    Régression:
        Objectif: Prédire la valeur de la pollution (pm2.5) en fonction de la superficie et de la surface végétalisée.
        Algorithmes: Régression linéaire, Régression Ridge, Régression Lasso, Arbres de régression, Forêts aléatoires.
        Intérêt: Comprendre l'impact de la superficie et de la surface végétalisée sur la qualité de l'air.
    Classification:
        Objectif: Classer les quartiers en fonction d'un critère (par exemple, quartiers à faible, moyenne ou forte pollution).
        Algorithmes: Régression logistique, Arbres de décision, Forêts aléatoires, SVM.
        Intérêt: Créer un modèle prédictif pour identifier les quartiers à risque en termes de pollution.

Guidage dans vos Choix

1. Définir votre objectif principal:

    Comprendre les relations: Si vous cherchez à comprendre les relations entre les variables, l'analyse en composantes principales et la régression linéaire sont de bons choix.
    Prédire la pollution: Pour prédire la pollution, la régression (linéaire, non-linéaire) est idéale.
    Identifier des groupes de quartiers: Le clustering vous permettra de regrouper les quartiers en fonction de leurs caractéristiques.

2. Préparer vos données:

    Nettoyage: Vérifier la présence de valeurs manquantes, d'outliers.
    Normalisation: Normaliser les variables si nécessaire (par exemple, si les échelles sont très différentes).
    Encodage: Si vous avez des variables catégorielles (par exemple, le nom de la ville), il faudra les encoder.

3. Choisir les bons algorithmes:

    Expérimenter: Tester différents algorithmes pour voir lequel donne les meilleurs résultats.
    Évaluer les performances: Utiliser des métriques appropriées (RMSE, MAE, précision, rappel, F1-score).
    Considérer la complexité: Pour les grands datasets, les algorithmes plus rapides (comme les arbres de décision ou les forêts aléatoires) peuvent être préférables.

4. Interpréter les résultats:

    Visualiser: Utiliser des graphiques pour mieux comprendre les résultats.
    Valider: Valider les modèles sur un jeu de données de test.
