# Guide d'installation

Pour utiliser les outils disponibles dans ce dépôt GitHub, vous aurez au préalable besoin d'installer quelques bibliothèques et autres outils. Vous trouverez par la suite un petit guide qui explique comment installer tout ce qui est nécessaire pour utiliser le notebook d'inférence, celui d'entraînement et même le script `datasetFormator.py`, qui permet de générer un dataset pour l'entraînement. Ce guide est écrit principalement pour une installation sous Windows (10 de préférence). Si vous utilisez un autre système d'exploitation (MacOs, Linux...), vous aurez peut-être besoin de chercher sur internet comment installer CUDA Toolkit, cuDNN et comment pouvoir lancer facilement les outils en créant un raccourci pour votre système.

## Sommaire
1. [Obtenir tous les fichiers nécessaires aux outils](#1-obtenir-tous-les-fichiers-nécessaires-aux-outils)
2. [Mettre en place l'environnement Python](#2-mettre-en-place-lenvironnement-python)
3. [Installer CUDA Toolkit et cuDNN](#3-installer-cuda-toolkit-et-cudnn)
4. [Installer l'API de Détection d'Objets de TensorFlow](#4-installer-lapi-de-détection-dobjets-de-tensorflow)
5. [Créer un raccourci pour ouvrir facilement les outils](#4-créer-un-raccourci-pour-ouvrir-facilement-les-outils)

## 1. Obtenir tous les fichiers nécessaires aux outils
1. [Télécharger](../archive/refs/heads/master.zip) ou cloner le [dépôt](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition.git).
2. Décompresser ou déplacer le dossier à l'emplacement de votre choix.
3. Télécharger le fichier de poids (`.h5`) et peut-être aussi une image ou plus sur lesquelles vous voulez lancer l'inférence. Placer ces fichiers dans le même dossier que précédemment, les images doivent être placées dans un dossier nommé `images` (cela peut être personnalisé dans le notebook d'inférence).

## 2. Mettre en place l'environnement Python
1. Télécharger et installer [MiniConda3](https://conda.io/en/latest/miniconda), vous pouvez aussi utiliser [Anaconda3](https://www.anaconda.com/products/individual#Downloads) (plus lourd).
2. Démarrer **Anaconda Prompt** en utilisant le **Menu Démarrer** ou la **Barre de Recherche Windows**.  
3. En utilisant la console, se déplacer dans le même dossier qu'à l'étape 2. 
    * Pour changer de dossier, utiliser la commande `cd <Nom du dossier>`.
    * Pour changer de disque utilisé, écrire la lettre du disque suivi de ":" et appuyer sur ENTRÉE (par exemple, pour passer du disque C au disque D, il faut écrire `D:` et appuyer sur ENTRÉE).  
4. Exécuter la commande suivante : `conda env create -f environment.yml`.

## 3. Installer CUDA Toolkit et cuDNN
En utilisant une carte graphique compatible avec CUDA 11.2, l'entraînement et les inférences peuvent être considérablement accélérés. Pour savoir si votre carte graphique est compatible, veuillez vous référer à la [liste des GPUs compatibles avec CUDA](https://developer.nvidia.com/cuda-gpus) pour vérifier que votre carte graphique a une `Compute Capability` d'au moins 3.5. Si c'est le cas, vous pouvez installer CUDA comme suit : 

1. Télécharger et installer [CUDA Toolkit 11.2.x](https://developer.nvidia.com/cuda-toolkit-archive) (pour TF 2.5).
2. Télécharger et installer [cuDNN 8.1.1 for CUDA 11.2](https://developer.nvidia.com/rdp/cudnn-archive) ([Guide d'installation [EN]](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)).


## 4. Installer l'API de Détection d'Objets de TensorFlow
:information_source: Cette étape n'est requise que si vous souhaitez entraîner Mask R-CNN Inception ResNet V2. L'entraînement requiert une carte graphique haut-de-gamme récente avec beaucoup de mémoire vidéo (VRAM), une GTX 1080 Ti (11 Go de VRAM) ayant déjà du mal à entraîner le réseau avec un batch de 1.

Pour entraîner Mask R-CNN Inception ResNet V2 de l'API de détection d'objets de TensorFlow (TF OD API), vous devez d'abord l'installer. Pour cela : 

1. Cloner ou télécharger et dézipper le dépôt GitHub [tensorflow/models](https://github.com/tensorflow/models) 

2. Télécharger et dézipper [Mask R-CNN Inception ResNet V2 1024x1024](http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz) de [TF2 Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) proche du dépôt précédent

3. En utilisant un nouvel environnement anaconda, se placer dans le dossier `models/research` 

4. Compiler les fichier Protobuf Compiler (protoc) :
    * Si protoc n'est pas installer, exécuter la commande suivante : `pip install protobuf`
    * Depuis le terminal, exécuter la commande suivante : `protoc object_detection/protos/*.proto --python_out=.`

5. Après cela, exécuter les lignes suivantes:

    ```shell
    cp object_detection/packages/tf2/setup.py .
    python -m pip install .
    ```
    Cette étape peut causer de nombreuses erreurs d'installation, il sera peut-être nécessaire d'utiliser la version de Python conseillée sur la [documentation de la TF OD API](https://github.com/tensorflow/models/blob/master/research/object_detection/README.md).

6. Pour tester si l'installation s'est correctement réalisée, exécuter la commande  `python object_detection/builders/model_builder_tf2_test.py` et observer si les tests ont réussi ou non. Vous devriez pour savoir dès cette étape si la carte graphique est utilisée. Pour cela, en utilisant le Gestionnaire des Tâches sur Windows (`Onglet Performance` > `GPU` > si la mémoire du GPU dédiée est utilisée, la carte graphique l'est aussi) ou en utilisant l'outil `nvidia-smi` depuis un terminal (vérifier la présence d'un processus python utilisant de la VRAM).

Source : [Creating your own object detector, Towards data science, Gilbert Tanner](https://towardsdatascience.com/creating-your-own-object-detector-ad69dda69c85)

## 5. Créer un raccourci pour ouvrir facilement les outils
1. En utilisant le **Menu Démarrer** ou la **Barre de Recherche Windows**, faire un clic-droit sur **Anaconda Prompt** et cliquer sur `Ouvrir l'emplacement du fichier`.
2. Dans l'explorateur de fichier, faire un clic-droit sur le raccourci **Anaconda Prompt** et cliquer sur `Envoyer vers`, et cliquer finalement sur `Bureau (créer un raccourci)`.
3. Accéder au Bureau. Déplacer le raccourci où bon vous semble, il sera utilisé pour accéder aux outils. Il est recommandé de le placer dans un endroit facile d'accès.
4. Faire un clic-droit sur le raccourci et cliquer sur `Propriétés`.
5. Définir la valeur du champ **Démarrer dans** comme le chemin d'accès du dossier où se trouve le dépôt téléchargé à la [première étape de ce guide d'installation](#1-obtenir-tous-les-fichiers-nécessaires-aux-outils).
6. Dans le champ **Cible**, remplacer `C:\Users\<UTILISATEUR>\miniconda3` avec `Skinet && jupyter notebook`.
7. (OPTIONNEL) Pour ouvrir directement un Notebook spécifique, ajouter le nom complet du fichier du Notebook (`.ipynb`) à la fin du champ **Cible**. La fin du champ devrait donc ressembler à `activate.bat Skinet && jupyter notebook MyNotebook.ipynb` par exemple.
8. (OPTIONNEL) L'icône du raccourci peut être changée pour quelque chose de plus adapté.
9. Cliquer sur le bouton `OK`

L'installation devrait désormais être terminée. Vous devriez tester le Notebook d'inférence pour être sûr que tout fonctionne correctement.