import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import ticker
from scipy.stats import levene
import os

# Palette pour stylisation des graphiques
PALETTE = [
    "#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51",
    "#6A4C93", "#1982C4", "#8AC926", "#FFCA3A", "#FF595E"
]


def apply_theme():
    ''' Applique un thème à tous les graphiques matplotlib.
     '''
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.edgecolor": "#444444",
        "axes.titlecolor": "#222",
        "legend.frameon": False,
        "figure.titlesize": 16,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    })


#
def annotater_bar(ax, fmt="{:,.0f}", spacing=3):
    '''
    Fonction pour annoter une barre dans un graphique à barres
    '''
    for p in ax.patches:
        if p.get_width() == 0 and p.get_height() == 0:
            continue
        # Détection orientation
        vertical = p.get_height() >= p.get_width()
        if vertical:
            value = p.get_height()
            ax.annotate(fmt.format(value),
                        (p.get_x() + p.get_width() / 2, value),
                        ha="center", va="bottom",
                        xytext=(0, spacing),
                        textcoords="offset points",
                        fontsize=9, fontweight="bold")
        else:
            value = p.get_width()
            ax.annotate(fmt.format(value),
                        (p.get_x() + value, p.get_y() + p.get_height() / 2),
                        ha="left", va="center",
                        xytext=(spacing, 0),
                        textcoords="offset points",
                        fontsize=9, fontweight="bold")


# Appliquer le thème pour tous les graphiques
apply_theme()


def save_fig(exercice: str, name: str, ext="png", dpi=150):
    """Sauvegarde le graphique avec un nom standardisé."""
    os.makedirs("figures", exist_ok=True)
    slug = name.lower().replace(" ", "_").replace("/", "_")
    ex_slug = exercice.replace(".", "_")
    fname = f"figures/ex_{ex_slug}_{slug}.{ext}"
    plt.savefig(fname, dpi=dpi, bbox_inches="tight")


def make_autopct_formatter(values, unit="$"):
    """
    Fonction pour formatter la valeur et le pourcentage dans un graphique en secteur (pie chart)
    """
    total = sum(values)

    def autopct_format(pct):
        valeur = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n({unit}{valeur:,})".replace(",", " ")

    return autopct_format


''''1.2 / 1.3) Configuration de l'affichage des DataFrames '''

pd.set_option('display.min_rows', 6)  # Minimum de lignes avant de trunquer
pd.set_option('display.max_columns', 12)  # Maximum de colonnes affichées
pd.set_option('display.max_rows', 100)  # Maximum de lignes affichées
pd.set_option('display.max_colwidth', 20)  # Maximum 20 caractères par colonne
pd.set_option('display.width', 240)  # Maximum 240 caractères par ligne

'''# Test de la configuration avec un DataFrame 20x200
def test_configuration():

    import numpy as np

    # Créer un DataFrame 20 colonnes x 200 lignes avec données variées
    np.random.seed(42)

    # Générer différents types de données
    data = {}
    for i in range(20):
        col_name = f'col_{i + 1:02d}_{"abcdefghijklmnopqrstuvwxyz"[i]}'

        if i % 4 == 0:  # Entiers
            data[col_name] = np.random.randint(0, 1000, 200)
        elif i % 4 == 1:  # Flottants
            data[col_name] = np.random.normal(100, 25, 200)
        elif i % 4 == 2:  # Chaînes longues pour tester max_colwidth
            data[col_name] = [f'texte_très_long_colonne_{i}_{j}' for j in range(200)]
        else:  # Booléens
            data[col_name] = np.random.choice([True, False], 200)

    df_large = pd.DataFrame(data)

    return df_large
'''

''' 1.4) Expliquez les concepts de base de Pandas, à savoir les DataFrames , les Series
et les Index

DATAFRAMES
Le DataFrame est la structure principale de Pandas pour manipuler des ensembles de données.
Il structuré de manière bidimensionnelles (2D), taille mutable et potentiellement contenir des données hétérogènes.

INDEX
La classe Index est un ensemble de valeurs qui est utilisée par Pandas pour étiqueter des colonnes d'un DataFrame ou des enregistrements d'une Series ou d'un DataFrame.

SERIES
La classe Series permet de gérer les données d'une tableau à une dimension (1D), semblable à une colonne dans une feuille de calcul.
Les valeurs d'une Serie, sont souvent associées à un Index, qui fait office "d'étiquette" de ligne.
'''

# Chargement des feuilles excels
orders = pd.read_excel('./CC-AIBD-DataAnalyse-donnees.xlsx', sheet_name='Sales Order_data')
territories = pd.read_excel('./CC-AIBD-DataAnalyse-donnees.xlsx', sheet_name='Sales Territory_data')
sales = pd.read_excel('./CC-AIBD-DataAnalyse-donnees.xlsx', sheet_name='Sales_data')
resellers = pd.read_excel('./CC-AIBD-DataAnalyse-donnees.xlsx', sheet_name='Reseller_data')
dates = pd.read_excel('./CC-AIBD-DataAnalyse-donnees.xlsx', sheet_name='Date_data')
products = pd.read_excel('./CC-AIBD-DataAnalyse-donnees.xlsx', sheet_name='Product_data')
customers = pd.read_excel('./CC-AIBD-DataAnalyse-donnees.xlsx', sheet_name='Customer_data')

dataframes = {
    "Orders": orders,
    "Territories": territories,
    "Sales": sales,
    "Resellers": resellers,
    "Dates": dates,
    "Products": products,
    "Customers": customers
}

'''
2.2) Affichez les cinq premières lignes de chaque DataFrame et vérifiez-en également les types de colonnes.
for name, df in dataframes.items():
    print(f"\n{'=' * 40}")
    print(f"{name} DataFrame")
    print("=" * 40)

    # 5 premières lignes
    print(df.head(), "\n") 
    print(df.dtypes, "\n")    
'''

'''
2.3) Affichez des informations de base sur les DataFrame via les méthodes info()
et describe() pour obtenir des informations sur la taille, les types de données
et les valeurs manquantes dans chaque DataFrame'''

for name, df in dataframes.items():
    print(f"\n{'=' * 40}")
    print(f"{name} DataFrame")
    print("=" * 40)

    # Info générale (taille, types, valeurs manquantes)
    df.info()

    # Valeurs manquants
    print("\n--- Nombre de valeurs manquantes par colonne (donc nulle) ---")
    print(df.isnull().sum())

    print("\n--- Statistiques descriptives ---")
    print(df.describe(include="all"))  # inclut toutes les colonnes

'''2.4) Notez les colonnes qui ont un type qui semble incorrect vis-à-vis de leur usage

Pour Sales:
    - Unit Price Discount Pct: type int64 alors qu'il s'agit de deux décimales après la virgule, il devrait être float64
    - ShipDateKey(float64) alors que les autres keys sont des int64 y compris dans Dates

'''

'''2.5.1) Affichez des informations de distribution (dispersion) du coût standard des
produits du catalogue. Affichez la médiane, l'écart-type et la variance.
( Category ).'''

mean_251 = products["Standard Cost"].mean()
median_251 = products["Standard Cost"].median()
std_251 = products["Standard Cost"].std()
variance_251 = products["Standard Cost"].var()
print("2.5.1)")
print(f"Moyenne : {mean_251}")
print(f"Écart-type : {std_251}")
print(f"Variance : {variance_251}")
print(f"Médiane: {median_251}")

'''2.5.2) Calculez les mêmes informations, mais cette fois-ci par catégorie de produit'''

print("2.5.2)")
print(products.groupby("Category")["Standard Cost"].agg(
    Moyenne="mean",
    Médiane="median",
    EcartType="std",
    Variance="var"
))

''' 3.1) 
Assurez-vous dans tous les DataFrames qu'il n'existe aucune cellule laissée
vide. Si ce n'est pas le cas, affichez les identifiants des colonnes contenant des
valeurs vides. Remplacez également ces valeurs par une valeur de repli si
besoin.
'''
for name, df in dataframes.items():
    # print(f"\n{'=' * 40}")
    # print(f"{name} DataFrame")
    # print("=" * 40)

    # print(df.isnull().sum())  # combien il y a de valeur nulles avant remplacement
    colonnes_vides = df.columns[df.isnull().any()]
    # print(colonnes_vides.tolist()) # affiche les colonnes qui contiennent des valeurs nulles
    for colonne in colonnes_vides:
        # On remplace les valeurs en fonction du type de données
        if df[colonne].dtype == "int64" or df[colonne].dtype == "float64":
            df[colonne] = df[colonne].fillna(0)
        elif df[colonne].dtype == "object":
            df[colonne] = df[colonne].fillna("NA")
        else:
            df[colonne] = df[colonne].fillna("NA")
    # print(df.isnull().sum()) # combien il y a de valeur nulles après remplacement

'''3.2) Vérifiez dans les ventes que le total des ventes est correct, notamment lorsque
la valeur de remise (discount) est non nulle.'''

# On remplace les valeurs nulles par 0 avant le calcul
discount_pct = sales["Unit Price Discount Pct"].fillna(0)
prix_avec_discount = sales["Unit Price"] * (1 - discount_pct)
invalides = abs(sales["Sales Amount"] - (sales[
                                             "Order Quantity"] * prix_avec_discount)) > 0.03  # certaines lignes n'ont pas de réduction et pourtant le Sales Amount est supérieur de 0.01, 0.02 ou 0.03 au résultat de sales["Unit Price"] * sales["Order Quantity"]
print(sales[invalides])  # Présente les lignes dont Sales Amount n'est pas cohérent

'''3.3)
Créez une colonne Price ratio calculant le multiplicateur entre le prix
affiché et le coût du produit.
'''
products["Price ratio"] = round(products["List Price"] / products["Standard Cost"], 4)
print("3.3)")
print(products["Price ratio"])

'''3.4)
Créez une colonne Line number dans orders extrayant le numéro de ligne de
la colonne Sales Order Line . '''

orders["Line Number"] = orders["Sales Order Line"].str.split("-").str[1].str.strip().astype(int)
print("3.4)")
print(orders)

'''3.5)
Traitez et convertissez les colonnes qui ne semblent pas avoir un type de
données correct inféré pour l'analyse.'''

sales["Unit Price Discount Pct"] = sales["Unit Price Discount Pct"].astype("float64")
sales["ShipDateKey"] = sales["ShipDateKey"].astype("int64")

'''3.6)
Créez une colonne Margin représentant la marge entre les ventes et le coût
total.
'''

sales["Margin"] = round(sales["Sales Amount"] - sales["Total Product Cost"], 4)

'''4.1)
Affichez le total des ventes, réparti par territoire en utilisant des graphiques à
barres. L'intitulé des barres doit être horizontal, et représenter le Country du
territoire ainsi que la Region sur une seconde ligne. Sauvegardez le
diagramme dans un fichier au format PNG.'''

df = sales.merge(territories, on="SalesTerritoryKey")
grouped = df.groupby(["Country", "Region"])["Sales Amount"].sum().reset_index()
grouped["Label"] = grouped["Country"] + "\n" + grouped["Region"]

plt.figure(figsize=(18, 5))
colors_41 = [PALETTE[i % len(PALETTE)] for i in range(len(grouped))]
bars = plt.bar(grouped["Label"], grouped["Sales Amount"], color=colors_41, edgecolor="#333")
plt.title("Total des ventes par territoire")
plt.xlabel("Territoires (Pays + Région)")
plt.ylabel("Total des ventes ($)")
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
plt.xticks(fontweight="bold")
annotater_bar(plt.gca(), fmt="{:,.0f}")
plt.tight_layout()
save_fig("4.1", "ventes_par_territoire")
plt.show()

'''4.2)
Affichez le total des ventes par Model de produit via une visualisation dans un
graphique adapté (secteur ou barres). Sauvegardez également le diagramme
dans un fichier au format PNG.'''

df = sales.merge(products, on="ProductKey")
grouped = df.groupby("Model")["Sales Amount"].sum().reset_index()
plt.figure(figsize=(30, 18))
colors_42 = [PALETTE[i % len(PALETTE)] for i in range(len(grouped))]
plt.bar(grouped["Model"], grouped["Sales Amount"], color=colors_42, edgecolor="#333")
plt.title("Total des ventes par modèle")
plt.xlabel("Modèle")
plt.ylabel("Total des ventes ($)")
plt.xticks(rotation=90)
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
annotater_bar(plt.gca(), fmt="{:,.0f}", spacing=2)
plt.tight_layout()
save_fig("4.2", "ventes_par_modele")
plt.show()

'''4.3)
Explorez les ventes par jour et identifiez les tendances saisonnières
potentielles; créez des colonnes avec des moyennes de ventes lissées sur 7, et
sur 30 jours. Affichez des graphiques en lignes pour voir ces tendances.'''

sales_tendance = sales.merge(dates, left_on="OrderDateKey", right_on="DateKey")
# On aggrège les ventes par jour
sales_tendance = sales_tendance.groupby('Date')["Sales Amount"].sum().reset_index()
# Création des moyennes mobiles 7 et 30 jours
sales_tendance['MA7'] = sales_tendance['Sales Amount'].rolling(window=7, center=True).mean()
sales_tendance['MA30'] = sales_tendance['Sales Amount'].rolling(window=30, center=True).mean()
plt.figure(figsize=(14, 6))
plt.plot(sales_tendance["Date"], sales_tendance["Sales Amount"], label="Ventes quotidiennes", alpha=0.4,
         color=PALETTE[0])
plt.plot(sales_tendance["Date"], sales_tendance["MA7"], label="Moyenne mobile 7 j", linewidth=2, color=PALETTE[3])
plt.plot(sales_tendance["Date"], sales_tendance["MA30"], label="Moyenne mobile 30 j", linewidth=2, color=PALETTE[4])
plt.title("Tendances des ventes quotidiennes (moyennes mobiles 7j / 30j)")
plt.xlabel("Date")
plt.ylabel("Montant des ventes ($)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
save_fig("4.3", "tendance_ventes_journalieres")
plt.show()

'''4.4)
Générez des tableaux croisés dynamiques (pivot tables) pour analyser les
ventes par différentes dimensions (par exemple, somme des ventes par produit
et par territoire).'''

# On fusionne les ventes avec les infos produit et territoire
sales_merged_44 = (sales.merge(products[["ProductKey", "Product", "Category", "Subcategory"]],
                               on="ProductKey", how="left")
                   .merge(territories[["SalesTerritoryKey", "Region", "Country", "Group"]],
                          on="SalesTerritoryKey", how="left"))
# Ventes par produit et territoire
pivot_prod_terr = pd.pivot_table(
    sales_merged_44,
    values="Sales Amount",
    index="Product",
    columns="Region",
    aggfunc="sum",
    fill_value=0
)
print("Ventes par produit et par région")
print(pivot_prod_terr)

'''5.1) Affichez les informations statistiques classiques des ventes (moyenne,
médiane, écart-type, minimum, maximum, premier quartile et troisième
quartile) pour chaque territoire.'''

sales_merge_territoire_5_1 = sales.merge(territories, on="SalesTerritoryKey")
# Calcul des statistiques descriptives par territoire
sales_merge_territoire_5_1_stats = sales_merge_territoire_5_1.groupby("Region")["Sales Amount"].agg(
    moyenne="mean",
    mediane="median",
    ecart_type="std",
    minimum="min",
    maximum="max"
)
sales_merge_territoire_5_1_stats["Q1"] = sales_merge_territoire_5_1.groupby("Region")["Sales Amount"].quantile(0.25)
sales_merge_territoire_5_1_stats["Q3"] = sales_merge_territoire_5_1.groupby("Region")["Sales Amount"].quantile(0.75)
print("5.1)\r\n")
print(sales_merge_territoire_5_1_stats)

'''5.2) Calculez la corrélation entre différentes variables :
Entre les ventes totales et les profits. Entre le coût total et la marge.'''

sales["Profit"] = sales["Sales Amount"] - sales["Total Product Cost"]
# Corrélation entre ventes et profits
corr_ventes_profit = sales["Sales Amount"].corr(sales["Profit"])
# Corrélation entre coût total et marge
corr_cout_marge = sales["Total Product Cost"].corr(sales["Margin"])
print("5.2)\r\n")
print(corr_ventes_profit)
print("Corrélation entre ventes et profits :", corr_ventes_profit)
print("Corrélation entre coût total et marge :", corr_cout_marge)

'''5.3) Identifiez les territoires et produits les plus performants en termes de ventes.'''

sales_merged_53 = (sales.merge(products[["ProductKey", "Product", "Category", "Subcategory"]],
                               on="ProductKey", how="left")
                   .merge(territories[["SalesTerritoryKey", "Region", "Country", "Group"]],
                          on="SalesTerritoryKey", how="left"))
# Nb ventes par produit et territoire
pivot_prod_terr_53 = pd.pivot_table(
    sales_merged_53,
    values="Sales Amount",
    index="Product",
    columns="Region",
    aggfunc="count",
    fill_value=0
)
print("5.3)\r\n")
print("Nombre de ventes par produit et par région")
print(pivot_prod_terr_53.head(10))

'''5.4) Effectuez des tests statistiques pour vérifier les hypothèses, par exemple,
l'écart type des ventes par territoire.'''

sales_tests_hypotheses = sales.merge(territories, on="SalesTerritoryKey")
print("5.4)\r\n")
print("=== TEST D'ÉGALITÉ DES ÉCARTS-TYPES PAR TERRITOIRE ===\n")
# 1. Calcul des écarts-types par territoire
std_par_territoire = sales_tests_hypotheses.groupby("Region")["Sales Amount"].std()
print("Std par territoire :")
for region, std_value in std_par_territoire.items():
    print(f"  {region}: {std_value:,.2f}")
print(f"\n--- Test de Levene ---")
print("H0: Les écarts-types sont égaux entre tous les territoires")
print("H1: Au moins un écart-type est différent\n")
# Préparation des données par région
data_par_region = []
regions = sales_tests_hypotheses["Region"].unique()
for region in regions:
    data_region = sales_tests_hypotheses[sales_tests_hypotheses["Region"] == region]["Sales Amount"].dropna()
    data_par_region.append(data_region.values)
# Test levene
levene_stat, levene_p = levene(*data_par_region)
print(f"Statistique : {levene_stat:.4f}")
print(f"P-value : {levene_p:.4f}")
print(f"Seuil de signification (α) : 0.05")
# Conclusion
if levene_p < 0.05:
    print(f"\nCONCLUSION : On rejette H0 (p = {levene_p:.4f} < 0.05)")
    print("Les écarts-types sont significativement différents entre les territoires.")
else:
    print(f"\nCONCLUSION : On ne rejette pas H0 (p = {levene_p:.4f} ≥ 0.05)")
    print("Pas de différence significative entre les écarts-types des territoires.")
# Ratio max/min pour quantifier la différence
max_std = std_par_territoire.max()
min_std = std_par_territoire.min()
ratio = max_std / min_std

print(f"\nRatio écart-type max/min : {ratio:.2f}")
print(f"Territoire avec le plus grand écart-type : {std_par_territoire.idxmax()} ({max_std:,.2f})")
print(f"Territoire avec le plus petit écart-type : {std_par_territoire.idxmin()} ({min_std:,.2f})")

'''6.1) Affichez un nuage de points (scatter) pour explorer les relations entre les
variables (total des ventes en x, marge en y, et prix unitaire pour l'intensité des
points).'''

# Créer le scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    sales["Sales Amount"],
    sales["Margin"],
    c=sales["Unit Price"],
    cmap="viridis",
    alpha=0.65,
    edgecolor="#222",
    linewidth=0.3
)
plt.colorbar(scatter, label="Prix unitaire")
plt.title("Relation Ventes / Marge / Prix unitaire")
plt.xlabel("Ventes ($)")
plt.ylabel("Marge ($)")
plt.tight_layout()
save_fig("6.1", "scatter_ventes_marge_prix_unitaire")
plt.show()

'''6.2) Créez des diagrammes à moustache (box plot) pour analyser la distribution en
quartiles des ventes. Une boîte par territoire, valeurs de boîte par modèle de produit.

'''
print("6.2 et 6.3)")
sales_623 = (
    sales
    .merge(products, on="ProductKey")
    .merge(territories, on="SalesTerritoryKey")
)
# limiter aux 10 (ou n) modèles les plus vendus globalement
top_models = (
    sales_623.groupby("Model")["Sales Amount"].sum().nlargest(10).index
)
filtered = sales_623[sales_623["Model"].isin(top_models)]
fig, ax = plt.subplots(figsize=(20, 12))
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
filtered.boxplot(column="Sales Amount", by=["Region", "Model"], ax=ax, rot=90,
                 boxprops=dict(color=PALETTE[0]),
                 medianprops=dict(color=PALETTE[4], linewidth=2),
                 whiskerprops=dict(color="#555"),
                 capprops=dict(color="#555"))
plt.title("Distribution des ventes par territoire et modèle")
plt.suptitle("")
plt.xlabel("Région / Modèle")
plt.ylabel("Montant des ventes ($)")
plt.tight_layout()
save_fig("6.2", "boxplot_ventes_par_territoire_modele")
plt.show()

'''6.4) Affichez une heat map (carte de chaleur ou matrice de corrélation) pour
visualiser les corrélations entre les variables :
Entre le total de vente et la marge'''

corr = sales[["Sales Amount", "Margin", "Total Product Cost"]].corr()
plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap="YlGnBu")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Matrice de corrélation (Ventes / Marge / Coûts)")
plt.tight_layout()
save_fig("6.4", "heatmap_correlation")
plt.show()

'''6.5) Créez des graphiques en secteur (pie) pour visualiser la part de marché des
différents produits ou territoires. Affichez la part en pourcent et/ou en dollars
dans les secteurs si possible.'''

sales_copy_6_5 = sales.copy()

sales_copy_6_5 = sales_copy_6_5.merge(territories, on="SalesTerritoryKey").merge(products, on="ProductKey")

ventes_par_territoire_65 = sales_copy_6_5.groupby("Region")["Sales Amount"].sum().sort_values(ascending=False)
ventes_par_produits_65 = sales_copy_6_5.groupby("Product")["Sales Amount"].sum().sort_values(ascending=False).head(
    20)  # nous limitons le résultat

plt.figure(figsize=(8, 8))
plt.pie(
    ventes_par_territoire_65.values,
    labels=ventes_par_territoire_65.index,
    autopct=make_autopct_formatter(ventes_par_territoire_65.values, unit="$"),
    colors=[PALETTE[i % len(PALETTE)] for i in range(len(ventes_par_territoire_65))]
)
plt.title("Répartition des ventes par territoire")
plt.tight_layout()
save_fig("6.5", "pie_ventes_territoires")
plt.show()
plt.figure(figsize=(10, 10))
plt.pie(
    ventes_par_produits_65.values,
    labels=ventes_par_produits_65.index,
    autopct=make_autopct_formatter(ventes_par_produits_65.values, unit="$"),
    colors=[PALETTE[i % len(PALETTE)] for i in range(len(ventes_par_produits_65))]
)
plt.title("Répartition des ventes (Top 20 produits)")
plt.tight_layout()
save_fig("6.5", "pie_ventes_top20_produits")
plt.show()

'''8.1) Fusionnez les différentes tables de données pour obtenir une vue complète
des ventes.'''

sales_copy = sales.copy()

complete_merged = sales_copy.merge(products, on="ProductKey")
complete_merged = complete_merged.merge(territories, on="SalesTerritoryKey")
complete_merged = complete_merged.merge(customers, on="CustomerKey")
complete_merged = complete_merged.merge(dates, left_on="OrderDateKey", right_on="DateKey")
complete_merged = complete_merged.merge(resellers, on="ResellerKey")

'''8.2) Vérifiez bien que toutes les lignes de la table des ventes sont présentes dans le
résultat.'''

print(complete_merged.head())
print(complete_merged.info())
print(len(complete_merged) == len(sales))

'''8.3) Réalisez une analyse croisée des données'''

# 1. Ventes par produit et par territoire
ventes_produit_territoire = pd.pivot_table(
    complete_merged,
    values="SalesOrderLineKey",
    index="Product",
    columns="Region",
    aggfunc="count",
    fill_value=0,
    sort=False
).head(20)
print("Ventes par produit et par territoire")
print(ventes_produit_territoire.head(), "\n")
plt.figure(figsize=(15, 30))
plt.imshow(ventes_produit_territoire, aspect="auto", cmap="Blues")
plt.colorbar(label="Nombre ventes")
plt.xticks(range(len(ventes_produit_territoire.columns)), ventes_produit_territoire.columns, rotation=45)
plt.yticks(range(len(ventes_produit_territoire.index)), ventes_produit_territoire.index)
plt.title("Ventes par produit et par territoire")
plt.xlabel("Territoire")
plt.ylabel("Produit")
# Ajoute les valeurs dans les cases
for i in range(len(ventes_produit_territoire.index)):
    for j in range(len(ventes_produit_territoire.columns)):
        text = plt.text(j, i, str(int(ventes_produit_territoire.iloc[i, j])),
                        ha="center", va="center", color="black", fontweight='bold')

plt.tight_layout()
save_fig("8.3", "heatmap_ventes_produit_territoire")
plt.show()

# 2. Nombre de produits par Model
nb_produits_modele = complete_merged.groupby("Model")["Product"].nunique().sort_values(ascending=False).head(20)

print("Nombre de produits par modèle")
print(nb_produits_modele.head(), "\n")
nb_produits_modele.plot(kind="barh", color="darkgreen", figsize=(8, 10))
plt.title("Nombre de produits par modèle")
plt.xlabel("Nombre de produits")
plt.ylabel("Modèle")
plt.tight_layout()
save_fig("8.3", "barh_nb_produits_par_modele")
plt.show()

# 3. Clients avec le plus de commandes
clients_plus_commandes = complete_merged[complete_merged["CustomerKey"] != -1].groupby("CustomerKey")[
    "SalesOrderLineKey"].nunique().sort_values(ascending=False).head(10)

print("Clients avec le plus de commandes")
print(clients_plus_commandes, "\n")
plt.figure(figsize=(10, 5))
clients_plus_commandes.plot(kind="barh", color="orange")
plt.title("Top 10 clients avec le plus de commandes")
plt.xlabel("Nombre de commandes")
plt.ylabel("Client")
plt.tight_layout()
save_fig("8.3", "barh_top_clients_commandes")
plt.show()

# 4. Clients ayant le plus dépensé
clients_plus_depense = complete_merged[complete_merged["CustomerKey"] != -1].groupby("CustomerKey")[
    "Sales Amount"].sum().sort_values(ascending=False).head(10)
print("Clients ayant le plus dépensé")
print(clients_plus_depense, "\n")
plt.figure(figsize=(10, 5))
clients_plus_depense.plot(kind="barh", color="green")
plt.title("Top 10 clients ayant le plus dépensé")
plt.xlabel("Client")
plt.ylabel("Montant total des ventes")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
save_fig("8.3", "barh_top_clients_depenses")
plt.show()

# 5. Nombre de ventes par catégorie de produit et par territoire
ventes_categorie_territoire = pd.pivot_table(
    complete_merged,
    values="SalesOrderLineKey",
    index="Category",
    columns="Region",
    aggfunc="count",
    fill_value=0
)
print("Nombre de ventes par catégorie de produit par territoire")
print(ventes_categorie_territoire.head())
ventes_categorie_territoire.plot(kind="bar", stacked=True, figsize=(12, 6))
plt.title("Nombre de ventes par catégorie et par territoire")
plt.xlabel("Catégorie")
plt.ylabel("Nombre de ventes")
plt.legend(title="Territoire")
plt.xticks(rotation=45, ha="right")
for i, category in enumerate(ventes_categorie_territoire.index):
    y_offset = 0
    for j, region in enumerate(ventes_categorie_territoire.columns):
        value = ventes_categorie_territoire.iloc[i, j]
        if value > 0:
            plt.text(i, y_offset + value / 2, str(int(value)),
                     ha='center', va='center',
                     fontweight='bold', color='white')
        y_offset += value
plt.tight_layout()
save_fig("8.3", "stacked_ventes_categorie_territoire")
plt.show()

# 8.4) Complétez les visualisations précédentes pour représenter les ventes globales
# par différentes dimensions (produit, territoire, période).

ventes_territoire = complete_merged.groupby("Region")["Sales Amount"].sum()
ax = ventes_territoire.plot(kind="barh", figsize=(12, 8), color="orange")
ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.title("Ventes par région")
plt.ylabel("Région")
plt.xlabel("Ventes ($)")
annotater_bar(ax, fmt="{:,.0f}")
plt.tight_layout()
save_fig("8.4", "barh_ventes_par_region")
plt.show()

ventes_produit = complete_merged.groupby("Product")["Sales Amount"].sum().head(10)
ax = ventes_produit.plot(kind="barh", figsize=(12, 8), color="orange")
ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.title("Ventes par produit (Top 10)")
plt.ylabel("Produit")
plt.xlabel("Ventes ($)")
annotater_bar(ax, fmt="{:,.0f}")
plt.tight_layout()
save_fig("8.4", "barh_ventes_par_produit_top10")
plt.show()

ventes_temps = complete_merged.groupby("Fiscal Quarter")["Sales Amount"].sum()
ventes_temps.plot(kind="line", figsize=(12, 5))
plt.title("Ventes globales dans le temps")
plt.xlabel("Date")
plt.ylabel("Ventes ($)")
plt.grid(True)
plt.tight_layout()
save_fig("8.4", "line_ventes_trimestres")
plt.show()

'''9.1) Analysez les ventes par revendeur et identifiez les revendeurs les plus
performants.'''

best_sellers = complete_merged[complete_merged["ResellerKey"] != -1].groupby("Reseller")[
    "Sales Amount"].sum().sort_values(ascending=False).head(10)

'''9.2) Visualisez les performances des revendeurs avec des graphiques à barres et des diagrammes circulaires.'''
best_sellers.plot(kind="barh")
plt.title("Performance des vendeurs")
plt.xlabel("Reseller")
plt.ylabel("Performances")
plt.grid(True)
plt.tight_layout()
save_fig("9.2", "barh_top_revendeurs")
plt.show()

plt.figure(figsize=(10, 10))
plt.pie(
    best_sellers.values,
    labels=best_sellers.index,
    autopct=make_autopct_formatter(best_sellers.values, unit="$"),
    colors=[PALETTE[i % len(PALETTE)] for i in range(len(best_sellers))])
plt.title("Répartition ventes (Top 10 revendeurs)")
plt.tight_layout()
save_fig("9.2", "pie_top_revendeurs")
plt.show()

'''9.3) Comparez les performances des revendeurs sur différentes périodes'''

# Ventes par revendeur et par trimestre
ventes_par_revendeur_trim = (
    complete_merged[complete_merged["ResellerKey"] != -1]
    .groupby(["Reseller", "Fiscal Quarter"])["Sales Amount"]
    .sum()
    .reset_index()
)
pivot_trim = ventes_par_revendeur_trim.pivot(index="Reseller", columns="Fiscal Quarter", values="Sales Amount").fillna(
    0)
print(pivot_trim.head())
# Visualisation (top 5 revendeurs par ventes totales)
top5_resellers = pivot_trim.sum(axis=1).sort_values(ascending=False).head(5).index
pivot_trim.loc[top5_resellers].T.plot(kind="bar", figsize=(12, 6))
plt.title("Comparaison des performances des revendeurs par trimestre")
plt.xlabel("Trimestre fiscal")
plt.ylabel("Montant des ventes")
plt.legend(title="Revendeur")
annotater_bar(ax, fmt="{:,.0f}")
plt.tight_layout()
save_fig("9.3", "bar_revendeurs_par_trimestre")
plt.show()

# Récupération de l'année
complete_merged["Year"] = pd.to_datetime(complete_merged["Date"]).dt.year
# Ventes par revendeur et par année
ventes_par_revendeur_annee = (
    complete_merged[complete_merged["ResellerKey"] != -1]
    .groupby(["Reseller", "Year"])["Sales Amount"]
    .sum()
    .reset_index()
)
pivot_year = ventes_par_revendeur_annee.pivot(index="Reseller", columns="Year", values="Sales Amount").fillna(0)
print(pivot_year.head())
# Visualisation : évolution des 5 meilleurs revendeurs
top5_resellers_year = pivot_year.sum(axis=1).sort_values(ascending=False).head(5).index
pivot_year.loc[top5_resellers_year].T.plot(kind="line", marker="o", figsize=(12, 6),
                                           color=[PALETTE[i % len(PALETTE)] for i in range(len(top5_resellers_year))])
plt.title("Évolution annuelle des performances des revendeurs")
plt.xlabel("Année")
plt.ylabel("Ventes ($)")
plt.legend(title="Revendeur")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
save_fig("9.3", "line_evolution_annuelle_revendeurs")
plt.show()