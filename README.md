This folder contains Jupyter notebooks for Project Fletcher of the Metis Data Science Bootcamp - Chicago cohort 6.

The focus of this project was Unsupervised Learning - The focus was on Natural Language processing, Topic Modeling and Clustering, using modeling techniques such as Non-Matrix Factorization, Latent Dirichlet Allocation, DB-Scan, K-Means Clustering, Gaussian Mixture Models, Latent Semantic Analysis, as well as Principal Component Analysis (PCA) for dimensionality reduction. I used T-distributed Stochastic Neighborhood embedding, a method of visualizing high dimensional data, or T-SNE for short, to view the effectiveness of my modeling techniques. For future work, I plan to create a recommendation engine for the Neural Image Processing Systems papers that I used as my dataset using cosine similarity, the beginning stages of this process can be seen in the Non-Matrix Factorization notebook. I also plan to integrate the algorithm into an online application using flask and bootstrap.

# Objective

Using a kaggle dataset regarding papers from the Neural Image Processing Systems Conference (NIPS), create a topic model for these papers, and in the near future create a recommendation engine.


functions.py: This notebook contains functions used throughout notebooks 1-5.

#notebooks.

NIPS webscrape: Contains a code for scraping the Neural Image Processing website for all of their published papers.
#https://github.com/benhamner/nips-papers/blob/master/src/download_papers.py

1. EDA & NLP Cleaning: This notebook contains the code used to clean and vectorize data to be used in Natural Language processing as well as some preliminary Exploratory Data Analysis.


2. Non-Matrix Factorization: This notebook contains analysis done using NMF visualized with T-SNE, and inital steps for building recommendation engine. NMF ultimately gave the best result optimized with perplexity=800.


3. GMM & DB-Scan: This notebook contains analysis with Gaussian Mixture Models, and DB-Scan. Visualization with T-SNE.


4. K-Means: This notebook contains K-Means Clustering and T-SNE visualization.

5. PCA, LDA & LSA: This notebook contains analysis done with PCA, LDA & LSA, and T-SNE visualization.

#pickle files.
Contains files that are used in the notebooks.
