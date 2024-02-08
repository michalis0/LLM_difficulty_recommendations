# Leveraging Large Language Models for Foreign Language Learning: Difficulty Estimation, Topic Estimation and Recommendation

## Description 🦭

This project contains the code used for our research work described in the article [**Leveraging Large Language Models for Foreign Language Learning: Difficulty Estimation, Topic Estimation and Recommendation**). It contains all the code for our experiments as well as the notebooks used to generate the figures in the article. The aim of the project is to enable :
- The classification of French texts according to their difficulty,
- The classification of French texts according to their subject,
- The recommendation of French texts according to the two previous criteria.
Each of these objectives is dealt with and detailed in a separate folder.

## Installation 🐼

### Requirements 🐨

This project uses the following tools:
- [Pyenv](https://github.com/pyenv/pyenv-installer) : Python version manager
- [Poetry](https://python-poetry.org/docs/#installation) : Python package manager
Tools that you can install using the following commands:
```bash
> curl https://pyenv.run | bash
> curl -sSL https://install.python-poetry.org | python3 -
```

Some notebooks also require the following installations:
- MySQL: [Installation](https://dev.mysql.com/doc/mysql-installation-excerpt/5.7/en/)
- A username and password to connect to the [CURNAGL](https://wiki.unil.ch/ci/books/high-performance-computing-hpc/page/curnagl) cluster at the University of Lausanne used for training the heaviest models. However, this code can be run on a local machine without using the [**Slurmray**](https://github.com/hjamet/SLURM_RAY) package.

### With Pyenv & Poetry 🐻

Once you have installed the above tools, you can install the project using the following commands:
```bash
> pyenv install 3.9.13 # Installs the version of Python used in the project
> pyenv local 3.9.13 # Defines the version of Python used in the project
> poetry install # Installs project dependencies
```

### Without Pyenv & Poetry 🐙

If you don't want to use Pyenv and Poetry, you can install the project dependencies using the following command:
```bash
> python -m venv .venv # Creates a virtual environment
> source .venv/bin/activate # Activates the virtual environment
> pip install -r requirements.txt # Installs project dependencies
```

## Repository Architecture 🦥

The project is structured as follows:
```
.
├── difficulty_estimation (8 notebooks for difficulty estimation)
├── topic_classification (5 notebooks for topic classification)
├── recommendation
├── src (Some frequently used functions and classes)
├── figures (The figures generated by the notebooks and used in the article)
├── README.md
├── requirements.txt
├── pyproject.toml
└── poetry.lock
```

Notebooks are intended to be executed in alphabetical order. Notebooks in the **"difficulty_estimation "** section are to be executed first, then those in the **"topic_classification "** section and finally those in the **"recommendation "** section.

*Note that executing certain notebooks may cause the following folders to appear:*
```
.
├── data (For data storage)
│ ├── raw
│ ├── processed
├── results (For storing results)
├── scratch (For storing temporary data)
├── .slogs (If using Slurmray)
```

### Difficulty Estimation 🐳

1. `a_Datasets` : This notebook allows you to download the data used to train the models. It also prepares the json files needed to train the OpenAI models and provides a cost estimate.
2. `b_OpenAiModelsTraining` : This notebook allows you to train the OpenAI models.
3. `c_OpenAiEvaluation` : This generates the results and metrics for the OpenAI models.
4. `d_OpenSourceModelsTraining` : This notebook is responsible for training the **CamemBERT** and **Mistral** models.
5. `e_OpenSourceEvaluation` : This generates the results and metrics for the **CamemBERT** and **Mistral** models.
6. `f_PairwiseMismatch` : This notebook trains and evaluates the performance of the **ARI**, **GFI** and **FKGL** Readability Indices. It also introduces the **Pairwise Mismatch** metric and evaluates the performance of all previous models on this metric.
7. `g_CreateFigures` : This notebook generates the figures used in the article.

### Topic Classification 🐬

1. `a_DataPreparation` : Ce notebook se charge du télechargement et prétaitement des données utilisées pour l'entrainement et l'évaluation des modèles.
2. `b_Zero-Shot` : Ce notebook se charge de l'adaptation d'un modèle **FlauBERT** pré-entrainé à notre tâche de classification de sujets. Il évalue également les performances de ce modèle ainsi que celles du modèle Zero-Shot de **mDeBERTa**.
3. `c_OpenAiTopicClassification` : Ce notebook se charge de l'évaluation des performances zero-shot des modèles OpenAI sur notre tâche de classification de sujets.
4. `d_FlaubertFineTuned` : Ce notebook vise à entrainer un modèle **FlauBERT** sur notre tâche de classification de sujets. Il évalue également les performances de ce modèle.
5. `e_CreateFigures` : Ce notebook génère les figures utilisées dans l'article.

### Recommendations 🐠

1. `a_GetZeeguData`: This notebook prepares the Zeegu dataset for recommendation.
2. `b_GenerateEmbeddings`: This notebook generates the ADA and BERT embeddings for all the datasets.
3. `c_Recommendations`: This notebook apply the recommender systems on all datasets. 
4. `d_Graphic`: This notebook generates a graph depicting how the performance of LightGCN evolves as the number of layers increases, specifically for the Zeegu dataset.
