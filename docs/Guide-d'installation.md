Pour utiliser les outils disponibles dans ce dépôt GitHub, vous aurez au préalable besoin d'installer quelques bibliothèques et autres outils. Vous trouverez par la suite un petit guide qui explique comment installer tout ce qui est nécessaire pour utiliser le notebook d'inférence, celui d'entraînement et même le script `datasetFormator.py`, qui permet de générer un dataset pour l'entraînement. Ce guide est écrit principalement pour une installation sous Windows (10 de préférence). Si vous utilisez un autre système d'exploitation (MacOs, Linux...), vous aurez peut-être besoin de chercher sur internet comment installer CUDA Toolkit, cuDNN et comment pouvoir lancer facilement les outils en créant un raccourci pour votre système.

# Sommaire
1. [Obtenir tous les fichiers nécessaires aux outils](#1-obtenir-tous-les-fichiers-nécessaires-aux-outils)
2. [Mettre en place l'environnement Python](#2-mettre-en-place-lenvironnement-python)
3. [Installer CUDA Toolkit et cuDNN](#3-installer-cuda-toolkit-et-cudnn)
4. [Créer un raccourci pour ouvrir facilement les outils](#4-créer-un-raccourci-pour-ouvrir-facilement-les-outils)

# 1. Obtenir tous les fichiers nécessaires aux outils
1. [Télécharger](../../../archive/refs/heads/Matterport_based.zip) ou cloner le [dépôt](../../../tree/Matterport_based).
2. Décompresser ou déplacer le dossier à l'emplacement de votre choix.
3. Télécharger le fichier de poids (`.h5`) et peut-être aussi une image ou plus sur lesquelles vous voulez lancer l'inférence. Placer ces fichiers dans le même dossier que précédemment, les images doivent être placées dans un dossier nommé `images` (cela peut être personnalisé dans le notebook d'inférence).

# 2. Mettre en place l'environnement Python
1. Télécharger et installer [MiniConda3](https://conda.io/en/latest/miniconda), vous pouvez aussi utiliser [Anaconda3](https://www.anaconda.com/products/individual#Downloads) (plus lourd).
2. Démarrer **Anaconda Prompt** en utilisant le **Menu Démarrer** ou la **Barre de Recherche Windows**.  
3. En utilisant la console, se déplacer dans le même dossier qu'à l'étape 2. 
    * Pour changer de dossier, utiliser la commande `cd <Nom du dossier>`.
    * Pour changer de disque utilisé, écrire la lettre du disque suivi de ":" et appuyer sur ENTRÉE (par exemple, pour passer du disque C au disque D, il faut écrire `D:` et appuyer sur ENTRÉE).  
4. Exécuter la commande suivante : `conda env create -f environment.yml`.

# 3. Installer CUDA Toolkit et cuDNN
1. Télécharger et installer [CUDA Toolkit 9.0](https://developer.nvidia.com/cuda-90-download-archive).
2. Télécharger et installer [cuDNN 7.0.5 pour CUDA 9.0](https://developer.nvidia.com/rdp/cudnn-archive) ([Guide d'installation [EN]](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)).

# 4. Créer un raccourci pour ouvrir facilement les outils
1. En utilisant le **Menu Démarrer** ou la **Barre de Recherche Windows**, faire un clic-droit sur **Anaconda Prompt** et cliquer sur `Ouvrir l'emplacement du fichier`.
2. Dans l'explorateur de fichier, faire un clic-droit sur le raccourci **Anaconda Prompt** et cliquer sur `Envoyer vers`, et cliquer finalement sur `Bureau (créer un raccourci)`.
3. Accéder au Bureau. Déplacer le raccourci où bon vous semble, il sera utilisé pour accéder aux outils. Il est recommandé de le placer dans un endroit facile d'accès.
4. Faire un clic-droit sur le raccourci et cliquer sur `Propriétés`.
5. Définir la valeur du champ **Démarrer dans** comme le chemin d'accès du dossier où se trouve le dépôt téléchargé à la [première étape de ce guide d'installation](#1-obtenir-tous-les-fichiers-nécessaires-aux-outils).
6. Dans le champ **Cible**, remplacer `C:\Users\<UTILISATEUR>\miniconda3` avec `Skinet && jupyter notebook`.
7. (OPTIONNEL) Pour ouvrir directement un Notebook spécifique, ajouter le nom complet du fichier du Notebook (`.ipynb`) à la fin du champ **Cible**. La fin du champ devrait donc ressembler à `activate.bat Skinet && jupyter notebook MyNotebook.ipynb` par exemple.
8. (OPTIONNEL) L'icône du raccourci peut être changée pour quelque chose de plus adapté.
9. Cliquer sur le bouton `OK`.

L'installation devrait désormais être terminée. Vous devriez tester le Notebook d'inférence pour être sûr que tout fonctionne correctement.
