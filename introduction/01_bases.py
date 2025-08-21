import pandas as pd


# Configuration de l'affichage des DataFrames
def configure_pandas_display():
    """
    Configure l'affichage par défaut des DataFrames pandas selon les spécifications :
    - Minimum 6 colonnes affichées
    - Maximum 12 colonnes affichées
    - Maximum 100 lignes affichées
    - Maximum 20 caractères par colonne
    - Maximum 240 caractères par ligne
    """

    # Nombre de colonnes à afficher
    pd.set_option('display.min_rows', 6)  # Minimum de lignes avant de trunquer
    pd.set_option('display.max_columns', 12)  # Maximum de colonnes affichées

    # Nombre de lignes à afficher
    pd.set_option('display.max_rows', 100)  # Maximum de lignes affichées

    # Largeur des colonnes et des lignes
    pd.set_option('display.max_colwidth', 20)  # Maximum 20 caractères par colonne
    pd.set_option('display.width', 240)  # Maximum 240 caractères par ligne

# Appliquer la configuration
configure_pandas_display()


# Test de la configuration avec un DataFrame 20x200
def test_configuration():
    """
    Crée un DataFrame de test avec 20 colonnes et 200 lignes
    pour vérifier que la configuration d'affichage fonctionne correctement
    """

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

# Concepts de base


''' DATAFRAMES
Le DataFrame est la structure principale de Pandas pour manipuler des ensembles de données.
Il structuré de manière bidimensionnelles (2D), taille mutable et potentiellement contenir des données hétérogènes.
'''

''' INDEX
La classe Index est un ensemble de valeurs qui est utilisée par Pandas pour étiqueter des colonnes d'un DataFrame ou des enregistrements d'une Series ou d'un DataFrame.
'''

''' SERIES
La classe Series permet de gérer les données d'une tableau à une dimension (1D), semblable à une colonne dans une feuille de calcul.
Les valeurs d'une Serie, sont souvent associées à un Index, qui fait office "d'étiquette" de ligne.
'''

# Utilisation principale
if __name__ == "__main__":
    # Appliquer la configuration
    configure_pandas_display()

    # Tester avec un DataFrame
    df_test = test_configuration()