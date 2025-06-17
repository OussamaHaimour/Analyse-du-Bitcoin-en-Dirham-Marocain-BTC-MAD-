# IMPORT STREAMLIT EN PREMIER (RIEN AVANT CETTE LIGNE)
import streamlit as st

# CONFIGURATION DE LA PAGE (DOIT ÊTRE IMMÉDIATEMENT APRÈS L'IMPORT)
st.set_page_config(layout="wide", page_title="Analyse Bitcoin")

# MAINTENANT LES AUTRES IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro

import io



# CHARGEMENT DES DONNÉES
@st.cache_data
def load_data():
    data = pd.read_csv('bitcoin_data.csv')
    data3 = data.copy()

    data1 = data.rename(columns={
        'Price': 'date',
        'Adj Close': 'Clôture Ajustée',
        'Close': 'Fermeture',
        'High': 'Plus haut',
        'Low': 'Plus bas',
        'Open': 'Ouverture',
        'Volume': 'Volume'})

    data1.drop(data1.index[0:2], inplace=True)
    data1.reset_index(drop=True, inplace=True)

    columns_ = ['Clôture Ajustée', 'Fermeture', 'Plus haut', 'Plus bas', 'Ouverture', 'Volume']
    for col in columns_:
        data1[col] = pd.to_numeric(data1[col])

    data2 = data1.copy()
    data1['date'] = pd.to_datetime(data1['date'])
    data2.drop('date', axis=1, inplace=True)
    data1 = pd.concat([data1['date'], data2], axis=1)

    for col in columns_:
        data1[col + ' MAD'] = data1[col] * 10.8

    data1['year'] = data1['date'].dt.year
    data1['month'] = data1['date'].dt.month
    data1['Rendement'] = data1['Clôture Ajustée MAD'].pct_change() * 100
    data1 = data1.dropna(subset=['Rendement'])
    data1['MMC'] = data1['Clôture Ajustée MAD'].rolling(window=20).mean()
    data1['MML'] = data1['Clôture Ajustée MAD'].rolling(window=50).mean()
    data1['Signal'] = 0
    data1.loc[data1['MMC'] > data1['MML'], 'Signal'] = 1
    data1.loc[data1['MMC'] < data1['MML'], 'Signal'] = -1

    return data1, data3


# INITIALISATION
data1, data3 = load_data()


# Fonctions de visualisation
def evolution_prix_cloture():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data1['date'], data1['Clôture Ajustée MAD'], label="Clôture Ajustée (MAD)", color='blue')
    ax.set_title("Évolution du Prix de Clôture Ajustée du Bitcoin (MAD)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix (MAD)")
    ax.legend()
    st.pyplot(fig)


def evolution_rendements():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data1['date'], data1['Rendement'], label="Rendement Quotidien", color='green', linewidth=1)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_title("Évolution des Rendements Quotidiens", fontsize=16)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Rendement (%)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    st.pyplot(fig)


def rendements_annuels():
    years = sorted(data1['year'].unique())
    cols = 2
    rows = -(-len(years) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(18, 10), sharex=False, sharey=False)
    axes = axes.flatten()

    for i, year in enumerate(years):
        yearly_data = data1[data1['year'] == year]
        axes[i].plot(yearly_data['date'], yearly_data['Rendement'], label=f"Rendement {year}", color='green',
                     linewidth=1)
        axes[i].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[i].set_title(f"Rendements Quotidiens - {year}", fontsize=12)
        axes[i].set_xlabel("Date")
        axes[i].set_ylabel("Rendement (%)")
        axes[i].legend()
        axes[i].grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig)


def statistiques_rendements():
    stats = {
        "Moyenne": data1['Rendement'].mean(),
        "Médiane": data1['Rendement'].median(),
        "Écart-type": data1['Rendement'].std(),
        "Variance": data1['Rendement'].var(),
        "Asymétrie (Skewness)": data1['Rendement'].skew(),
        "Aplatissement (Kurtosis)": data1['Rendement'].kurt()
    }
    st.table(pd.DataFrame(stats.items(), columns=["Statistique", "Valeur"]))


def histogramme_densite():
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data1['Rendement'], bins=50, kde=False, color='blue', label='Histogramme', ax=ax)
    sns.kdeplot(data1['Rendement'], color='red', linewidth=2, label='Densité', ax=ax)
    ax.set_title("Distribution des Rendements Quotidiens", fontsize=16)
    ax.set_xlabel("Rendements (%)", fontsize=14)
    ax.set_ylabel("Fréquence", fontsize=14)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


def evolution_prix_annuelle():
    data1['day_of_year'] = data1['date'].dt.dayofyear
    years = data1['year'].unique()

    fig, ax = plt.subplots(figsize=(14, 8))
    for year in years:
        yearly_data = data1[data1['year'] == year]
        ax.plot(yearly_data['day_of_year'], yearly_data['Clôture Ajustée MAD'], label=str(year), alpha=0.8)

    ax.set_title("Évolution du prix de clôture sur l'année (MAD)", fontsize=16)
    ax.set_xlabel("Jour de l'année", fontsize=14)
    ax.set_ylabel("Prix de clôture (MAD)", fontsize=14)
    ax.legend(title="Années", loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)


def relation_plus_bas_volume():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data1['Plus bas MAD'], data1['Volume'], color='blue', alpha=0.5)
    ax.set_title("Relation entre le prix le plus bas et le volume", fontsize=16)
    ax.set_xlabel("Prix le plus bas (MAD)", fontsize=14)
    ax.set_ylabel("Volume de trading", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

    correlation = np.corrcoef(data1['Plus bas MAD'], data1['Volume'])[0, 1]
    st.write(f"**Coefficient de corrélation de Pearson :** {correlation:.4f}")


def relation_plus_haut_volume():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data1['Plus haut MAD'], data1['Volume'], color='green', alpha=0.5)
    ax.set_title("Relation entre le prix le plus haut et le volume", fontsize=16)
    ax.set_xlabel("Prix le plus haut (MAD)", fontsize=14)
    ax.set_ylabel("Volume de trading", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

    correlation = np.corrcoef(data1['Plus haut MAD'], data1['Volume'])[0, 1]
    st.write(f"**Coefficient de corrélation de Pearson :** {correlation:.4f}")


def test_normalite():
    stat, p_value = shapiro(data1['Rendement'])
    alpha = 0.05
    if p_value > alpha:
        st.success("Les rendements suivent probablement une distribution normale.")
    else:
        st.error("Les rendements ne suivent pas une distribution normale.")
    st.write(f"**Statistique de test :** {stat:.4f}")
    st.write(f"**p-valeur :** {p_value:.4f}")


def autocorrelation():
    auto_corr_k1 = data1['Rendement'].autocorr(lag=1)
    auto_corr_k2 = data1['Rendement'].autocorr(lag=2)
    auto_corr_k3 = data1['Rendement'].autocorr(lag=3)

    st.write(f"**Auto-corrélation pour k=1 :** {auto_corr_k1:.4f}")
    st.write(f"**Auto-corrélation pour k=2 :** {auto_corr_k2:.4f}")
    st.write(f"**Auto-corrélation pour k=3 :** {auto_corr_k3:.4f}")


def graphique_autocorrelation():
    fig, ax = plt.subplots(figsize=(10, 6))
    sm.graphics.tsa.plot_acf(data1['Rendement'], lags=50, alpha=0.05, ax=ax)
    ax.set_title("Autocorrélogramme des Rendements Quotidiens", fontsize=16)
    ax.set_xlabel("Décalage (Lags)", fontsize=14)
    ax.set_ylabel("Auto-corrélation", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)


def graphique_moyennes_mobiles():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data1['date'], data1['Clôture Ajustée MAD'], label='Prix de Clôture', color='blue')
    ax.plot(data1['date'], data1['MMC'], label='MMC (20 jours)', color='green')
    ax.plot(data1['date'], data1['MML'], label='MML (50 jours)', color='red')

    buy_signals = data1[data1['Signal'] == 1]
    sell_signals = data1[data1['Signal'] == -1]

    ax.scatter(buy_signals['date'], buy_signals['Clôture Ajustée MAD'], color='green', marker='^', alpha=1,
               label='Acheter')
    ax.scatter(sell_signals['date'], sell_signals['Clôture Ajustée MAD'], color='red', marker='v', alpha=1,
               label='Vendre')

    ax.set_title("Prix de Clôture avec Signaux d'Achat/Vente", fontsize=16)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Prix de Clôture (MAD)", fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)


def strategie_moyennes_mobiles():
    cash = 1_000_000
    btc = 0

    for i in range(1, len(data1)):
        signal = data1['Signal'].iloc[i]
        price = data1['Clôture Ajustée MAD'].iloc[i]

        if signal == 1:
            cash -= 0.1 * price
            btc += 0.1
        elif signal == -1:
            cash += 0.1 * price
            btc -= 0.1

    dernier_prix = data1['Clôture Ajustée MAD'].iloc[-1]
    valeur_finale = cash + (btc * dernier_prix)
    rendement_total = ((valeur_finale / 1_000_000) - 1) * 100

    date_debut = data1['date'].iloc[0]
    date_fin = data1['date'].iloc[-1]
    n_years = (date_fin - date_debut).days / 365
    rendement_annualise = ((valeur_finale / 1_000_000) ** (1 / n_years) - 1) * 100

    st.write(f"**Valeur finale du portefeuille :** {valeur_finale:,.2f} MAD")
    st.write(f"**Rendement total :** {rendement_total:.2f}%")
    st.write(f"**Rendement annualisé :** {rendement_annualise:.2f}%")


def tester_strategie_annee(annee, strategie):
    data_annee = data1[data1['year'] == annee].copy()
    if len(data_annee) < 2:
        st.warning(f"Pas assez de données pour l'année {annee}.")
        return

    if strategie == "Moyennes Mobiles":
        cash = 1_000_000
        btc = 0

        for i in range(1, len(data_annee)):
            signal = data_annee['Signal'].iloc[i]
            prix = data_annee['Clôture Ajustée MAD'].iloc[i]

            if signal == 1:
                cash -= 0.1 * prix
                btc += 0.1
            elif signal == -1:
                cash += 0.1 * prix
                btc -= 0.1

        prix_final = data_annee['Clôture Ajustée MAD'].iloc[-1]
        valeur_finale = cash + (btc * prix_final)
        rendement = ((valeur_finale / 1_000_000) - 1) * 100

        st.write(f"**Stratégie :** Moyennes Mobiles")
        st.write(f"**Valeur finale :** {valeur_finale:,.2f} MAD")
        st.write(f"**Rendement :** {rendement:.2f}%")

    elif strategie == "Buy & Hold":
        prix_initial = data_annee['Clôture Ajustée MAD'].iloc[0]
        prix_final = data_annee['Clôture Ajustée MAD'].iloc[-1]
        btc = 1_000_000 / prix_initial
        valeur_finale = btc * prix_final
        rendement = ((valeur_finale / 1_000_000) - 1) * 100

        st.write(f"**Stratégie :** Buy & Hold")
        st.write(f"**Prix initial :** {prix_initial:,.2f} MAD")
        st.write(f"**Prix final :** {prix_final:,.2f} MAD")
        st.write(f"**Valeur finale :** {valeur_finale:,.2f} MAD")
        st.write(f"**Rendement :** {rendement:.2f}%")

    elif strategie == "Sans Risque":
        valeur_finale = 1_000_000 * 1.03
        rendement = 3.0

        st.write(f"**Stratégie :** Sans Risque (3%)")
        st.write(f"**Valeur finale :** {valeur_finale:,.2f} MAD")
        st.write(f"**Rendement :** {rendement:.2f}%")

# INTERFACE UTILISATEUR
def main():
    # Page d'accueil vide initialement
    if 'menu_principal' not in st.session_state:
        st.title("Analyse du Bitcoin (BTC-MAD)")
        st.markdown("""
        ## Bienvenue dans l'outil d'analyse du Bitcoin
        Veuillez sélectionner une option dans le menu de gauche.
        """)
        st.session_state.menu_principal = None

    # Menu sidebar
    with st.sidebar:
        st.header("Menu Principal")
        menu_principal = st.radio(
            "Options disponibles",
            ["Préparation des données", "Analyse détaillée", "Stratégies de trading"],
            index=None
        )
        if menu_principal:
            st.session_state.menu_principal = menu_principal

    # Contenu conditionnel
    if st.session_state.menu_principal == "Préparation des données":
        st.header("Préparation des données")
        option = st.selectbox("Choisir une option",
                              ["Structure des données originales",
                               "Structure des données transformées"])

        if option == "Structure des données originales":
            st.write("**Données originales :**")
            st.dataframe(data3.head())
            buffer = io.StringIO()
            data3.info(buf=buffer)
            st.text(buffer.getvalue())
            st.markdown("""
                    **Remarque :**

                    Les données affichées ne sont pas prêtes pour l'analyse, car :

                    - Les valeurs dans les colonnes sont au format **string**, alors qu'elles doivent être **numériques**.
                    - Le nom de la colonne des **dates est incorrect**.
                    - Les *deux premières **lignes** du fichier sont **vides**.

                    Veuillez effectuer les corrections nécessaires avant de procéder à une analyse plus approfondie.
                    """)

        elif option == "Structure des données transformées":
            st.write("**Données transformées :**")
            st.dataframe(data1.head())
            st.write("**Valeurs manquantes :**")
            st.write(data1.isna().sum())
            st.write("**Informations sur le DataFrame :**")
            buffer = io.StringIO()
            data1.info(buf=buffer)
            st.text(buffer.getvalue())
            st.markdown("""
                    ### Prétraitement des données et enrichissement

                    Le prétraitement des données a consisté en plusieurs étapes clés : vérification de l’intégrité des données, conversion des types pour faciliter la manipulation, et transformation des prix en dirhams marocains (MAD) afin de contextualiser localement l’analyse.

                    Des colonnes supplémentaires ont été ajoutées pour enrichir la base de données :

                    - **Année** et **Jour** pour permettre une analyse temporelle fine ;
                    - **MMC** (Moyenne Mobile Courte sur 20 jours) et **MML** (Moyenne Mobile Longue sur 50 jours) pour construire une stratégie de croisement ;
                    - **Signal** : une colonne générée automatiquement en fonction du croisement des moyennes mobiles.  
                      Un signal d’achat (+1) est déclenché lorsque la MMC croise la MML à la hausse, et un signal de vente (–1) lorsqu’elle la croise à la baisse.

                    Ces signaux sont ensuite utilisés pour simuler des opérations de trading et calculer les profits et pertes (P&L) basés sur les données historiques du Bitcoin.
                    """)

    elif st.session_state.menu_principal == "Analyse détaillée":
        st.header("Analyse détaillée des données")
        option = st.selectbox("Choisir une analyse",
                              ["Évolution du prix de clôture",
                               "Relation prix bas/volume et prix haut/volume",
                               "Rendements quotidiens",
                               "Statistiques des rendements",
                               "Distribution des rendements",
                               "Autocorrélation",
                               "Test de normalité"])

        if option == "Évolution du prix de clôture":
            evolution_prix_cloture()
            evolution_prix_annuelle()
            st.markdown("""
                        ### Analyse de l'évolution du prix du Bitcoin (2019 - 2025)

                        Le graphique retrace l’évolution du prix de clôture ajustée du **Bitcoin en dirham marocain (MAD)** entre **janvier 2019 et janvier 2025**. Malgré des **fluctuations marquées, une** tendance haussière à long terme se dégage.  
                        Le prix est passé d’environ *50 000 MAD* en 2019 à *1 000 000 MAD* fin 2025.

                        #### Faits marquants :

                        - **2020** : Hausse rapide jusqu’à 200 000 MAD, portée par la pandémie.
                        - **Avril 2021**: Pic historique à 600 000 MAD, suivi d’un effondrement à cause des régulations restrictives.
                        - **2022** : Déclin marqué.
                        - **Début 2023** : Reprise soutenue par l’adoption croissante de la blockchain et des investissements majeurs.

                        #### Tendances saisonnières observées :

                        - **Volatilité élevée** en début d’année.
                        - **Stabilité relative** en été.
                        - **Pics fréquents** en **janvier*, **mai** et **octobre**.

                        #### Facteurs influents :

                        - Internes : Mises à jour du réseau, évolution technologique.
                        - Externes : Politiques monétaires, régulations, tensions géopolitiques.

                        ---

                        **Conclusion :**
                        Le *Bitcoin* apparaît comme un *actif à fort potentiel mais à haut risque, dont l’analyse nécessite une **approche quantitative et contextuelle*.
                        """)

        elif option == "Relation prix bas/volume et prix haut/volume":
            relation_plus_bas_volume()
            relation_plus_haut_volume()
            st.markdown("""
                        ### Corrélation entre le volume de trading et les prix extrêmes du Bitcoin

                        L’analyse statistique révèle une **faible corrélation** entre le **volume de trading** du Bitcoin et ses **niveaux de prix extrêmes**, avec des coefficients de :

                        - **0,386** pour le **prix le plus bas**
                        - **0,414** pour le **prix le plus haut**

                        Ces résultats indiquent une *relation linéaire négligeable* entre ces variables.

                        Cette absence de lien significatif suggère que le volume de trading est probablement influencé par **d'autres facteurs** comme :

                        - Les **actualités économiques**
                        - Les **régulations**
                        - Le **comportement des investisseurs**

                        Pour une compréhension plus approfondie de cette dynamique, il serait pertinent d'intégrer **d'autres variables explicatives** telles que la **volatilité** ou des **indicateurs techniques**.
                        """)

        elif option == "Rendements quotidiens":
            evolution_rendements()
            rendements_annuels()
            st.markdown("""
            ### Analyse des rendements quotidiens du Bitcoin (2019 - 2024)

            Les graphiques des rendements quotidiens entre 2019 et 2024 révèlent des différences notables selon les années.  
            L’année 2020 se caractérise par une forte volatilité, probablement liée à la pandémie, avec des variations dépassant ±20 %.  
            À l’inverse, 2023 et 2024 montrent une relative stabilité, avec des fluctuations plus modérées autour de 0 %.  
            Globalement, les rendements restent majoritairement centrés sur la ligne zéro, ponctués de pics occasionnels.  
            Cette visualisation facilite l’identification des périodes de turbulence ou de calme, et soutient l’analyse des performances du marché.
            """)
        elif option == "Statistiques des rendements":
            statistiques_rendements()
            st.markdown("""
            ### Analyse descriptive des rendements journaliers du Bitcoin

            L’analyse descriptive des rendements journaliers du Bitcoin révèle une moyenne légèrement positive de 0,21 %, accompagnée d’une médiane plus faible, suggérant une distribution asymétrique.  
            La volatilité est élevée, avec un écart-type de 3,36 % et une kurtose de 10,14, indiquant une fréquence accrue d’événements extrêmes.  
            L’asymétrie négative (-0,26) souligne une tendance à des rendements négatifs plus marqués.  
            Ces caractéristiques traduisent un marché instable, où le risque et les fluctuations extrêmes doivent être pris en compte dans toute approche d’investissement.
            """)
        elif option == "Distribution des rendements":
            histogramme_densite()
            st.markdown("""
            ### Distribution des rendements quotidiens du Bitcoin (MAD)

            Le graphique présente la distribution des rendements quotidiens du Bitcoin (MAD) à travers un histogramme et une courbe de densité. On observe que :

            * La distribution se caractérise par des queues épaisses (à gauche et à droite), indiquant la présence d’événements extrêmes (rendements élevés ou pertes importantes) plus fréquents que dans une distribution normale.
            * La courbe de densité rouge montre une concentration des rendements autour de valeurs proches de zéro, avec des pics marqués, confirmant un kurtosis élevé (leptokurtique) mesuré précédemment (Kurtosis = 10.14).
            * L’histogramme bleu révèle une fréquence plus élevée de rendements négatifs que positifs, cohérente avec l’asymétrie négative (Skewness = -0.26).

            La distribution des rendements quotidiens du Bitcoin n’est pas normale, avec une tendance aux risques élevés en raison de la volatilité intense et de la fréquence des événements extrêmes. Ces caractéristiques reflètent la nature spéculative du marché et soulignent la nécessité d’utiliser des outils de gestion des risques avancés lors du trading.
            """)
        elif option == "Autocorrélation":
            autocorrelation()
            graphique_autocorrelation()
            st.markdown("""
            ### Auto-corrélation des rendements quotidiens du Bitcoin

            L’analyse de l’auto-corrélation des rendements quotidiens du Bitcoin pour des décalages k = 1, 2 et 3 révèle des valeurs proches de zéro, indiquant l’absence de relation significative avec les rendements passés.  
            Le graphique des lags jusqu’à 50 confirme cette faible auto-corrélation, avec la majorité des points dans l’intervalle de confiance.  
            Ces résultats soulignent l’imprévisibilité du marché du Bitcoin à court terme et la difficulté d’utiliser des modèles simples pour anticiper ses mouvements.
            """)

        elif option == "Test de normalité":
            test_normalite()
            st.markdown("""
            ### Test de normalité
            La conclusion principale est que la nature des rendements quotidiens du Bitcoin est instable et asymétrique, rendant les modèles statistiques traditionnels basés sur l’hypothèse de normalité inappropriés dans ce contexte.  
            Au lieu de cela, il est préférable d’utiliser des modèles plus flexibles qui prennent en compte les distributions à queues épaisses et la forte volatilité typique du marché des cryptomonnaies.
            """)

    elif st.session_state.menu_principal == "Stratégies de trading":
        st.header("Stratégies de trading")
        option = st.selectbox("Choisir une stratégie",
                              ["Moyennes mobiles",
                               "Comparaison par année"])

        if option == "Moyennes mobiles":
            graphique_moyennes_mobiles()
            st.subheader("Résultats de la stratégie")
            strategie_moyennes_mobiles()
            st.markdown("""
                    ### Stratégie des moyennes mobiles sur le Bitcoin (MAD)

                    Le graphique illustre l’application de la stratégie des moyennes mobiles sur le Bitcoin (MAD), où les croisements entre la moyenne mobile courte (MMC) et longue (MML) génèrent des signaux d’achat et de vente.  
                    Bien que la stratégie identifie correctement certaines tendances haussières, son efficacité reste variable dans un marché fortement volatil.  
                    Des ajustements, tels que l’intégration d’indicateurs de volatilité, de filtres de confirmation et de mécanismes de gestion des risques, sont recommandés pour en améliorer la fiabilité.

                    Les résultats de la simulation montrent un rendement total exceptionnel de 1536,38 %, avec une valeur finale de portefeuille de 16 363 788,73 MAD pour un investissement initial de 1 000 000 MAD, et un rendement annualisé de 59,34 %.  
                    Toutefois, des performances négatives comme en 2021 (-116,72 %) soulignent la sensibilité de la stratégie à des marchés instables et aux signaux retardés.  
                    Cela met en évidence la nécessité d’une adaptation dynamique et d’une diversification des outils d’analyse pour renforcer la robustesse de la stratégie face à l’incertitude du marché des cryptomonnaies.
                    """)

        elif option == "Comparaison par année":
            annee = st.selectbox("Choisir une année", sorted(data1['year'].unique()))
            strategie = st.selectbox("Choisir une stratégie",
                                     ["Moyennes Mobiles",
                                      "Buy & Hold",
                                      "Sans Risque"])

            if st.button("Exécuter la stratégie"):
                tester_strategie_annee(annee, strategie)
                st.markdown("""
                        ### Comparaison des stratégies d’investissement

                        La stratégie des moyennes mobiles appliquée au Bitcoin entre 2019 et 2025 montre un potentiel de rendement élevé, parfois supérieur à celui de la stratégie d’achat-conservation (Buy & Hold), surtout en période de tendance haussière.  
                        Toutefois, des performances annuelles variables révèlent des risques importants : en 2021, la stratégie de moyennes mobiles enregistré une perte de -116,72 %, tandis que le Buy & Hold affichait -63,21 % en 2022.  
                        Ces résultats contrastent avec la stabilité d’un investissement sans risque à 3 % annuel, soulignant le compromis entre rendement et sécurité.  
                        Ainsi, si les stratégies actives peuvent générer de forts gains, elles nécessitent une gestion rigoureuse des risques face à la volatilité du marché.
                        """)


# POINT D'ENTRÉE
if __name__ == "__main__":
    main()