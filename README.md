# TP : Chaînes de Markov et Modèles de Markov Cachés (HMM)

[cite_start]Ce projet met en œuvre des Chaînes de Markov et des HMM (Hidden Markov Models) pour deux applications distinctes : la modélisation de données météorologiques (pluie) et la reconnaissance vocale de mots isolés [cite: 64-66].

## 📂 Structure du Projet

Le TP est divisé en trois parties principales :

1.  [cite_start]**Météo (Chaîne de Markov Discrète)** : Modélisation de l'alternance entre périodes sèches et pluvieuses à l'aide d'une chaîne à 2 états ($E_0$=Sec, $E_1$=Pluie) [cite: 92-98].
2.  [cite_start]**Météo (HMM)** : Raffinement du modèle précédent en introduisant une couche cachée représentant l'état du ciel (Ciel clair, Nuageux, Très nuageux) pour prédire la pluie (observable binaire) [cite: 138-141].
3.  [cite_start]**Reconnaissance Vocale (HMM Gaussiens)** : Classification de mots isolés (ex: 'apple', 'banana') en utilisant des HMM à émission gaussienne sur différentes caractéristiques audio (Spectrum, Filter, MFCC) [cite: 164-172].

## 🛠 Installation et Prérequis

**Attention :** Ce projet dépend d'une ancienne version de la librairie `pomegranate`. La version 1.0+ n'est **pas** compatible avec le code fourni (`TpHmmUtilit.py`).

### Environnement recommandé
Il est conseillé d'utiliser un environnement virtuel (Python 3.9 - 3.11 recommandés).

```bash
# Création de l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows