#function to drop duplicates in dataframe.
def drop_duplicate_papers(df):
    return df.drop_duplicates(subset='id', keep='first')
#function to drop columns
def drop_columns(df, column):
    'drop columns'
    return df.drop(columns = column, inplace=True)

#document cleaning functions, tokenizing/stemming & creating tfidf matrices.
def clean(document):
    text = re.sub('[^a-zA-Z]+', ' ', document)
    for sign in ['\n', '\x0c']:
        text = text.replace(sign, ' ')
    return text.lower()
def tokenizing_function(document):
    stemmer = PorterStemmer()
    tokenized = [stemmer.stem(word) for sentence in sent_tokenize(document) for word in word_tokenize(sentence)]
    return tokenized
def create_tfidf(df, column_name, num_features, ngram=1):
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,ngram), stop_words='english', max_df=0.9,min_df=2,
                                       max_features=num_features, tokenizer=tokenizing_function)
    tfidf_sparse = tfidf_vectorizer.fit_transform(df[column_name])
    return tfidf_vectorizer, tfidf_sparse

#pickle functions
def pickle_(obj, name):
    with open(name+'.pickle', 'wb') as f:
        pickle.dump(obj, f, protocol=4)

def unpickle_(pkl):
    return pickle.load(open(pkl, 'rb'))

#check top words of each topic from nmf_tfidf
def topic_model(vectorizer, nmf_model, num_words):
    word_list = vectorizer.get_feature_names()
    components = nmf_model.components_
    for i in range(len(components)):
        top_words_index = components[i].argsort()[::-1][:num_words]
        top_words = [word_list[index] for index in top_words_index]
        print('Topic {}'.format(i+1))
        print(top_words)
#second function used for k-means as it has no attribute component, use clustercenters_ instead.
def topic_model2(vectorizer, model, num_words):
    word_list = vectorizer.get_feature_names()
    components = model.cluster_centers_
    for i in range(len(components)):
        top_words_index = components[i].argsort()[::-1][:num_words]
        top_words = [word_list[index] for index in top_words_index]
        print('Topic {}'.format(i+1))
        print(top_words)
#third function used for GMM as it has no attribute component, use means_ instead.
def topic_model3(vectorizer, model, num_words):
    word_list = vectorizer.get_feature_names()
    components = model.means_
    for i in range(len(components)):
        top_words_index = components[i].argsort()[::-1][:num_words]
        top_words = [word_list[index] for index in top_words_index]
        print('Topic {}'.format(i+1))
        print(top_words)

#function to scale a matrix using Standard Scaler
def scaled_matrix(matrix):
    scaler = StandardScaler().fit_transform(matrix)
    return scaler

def top_doc_per_topic(nmf_topic_space, num_docs, df):
    index_per_topic = nmf_topic_space.T.argsort(axis=1)
    for i, topic in enumerate(index_per_topic):
        top_indices = topic[-num_docs:][::-1]
        print('\nTopic {}'.format(i+1))
        for index in top_indices:
            title = df['title'][index]
            print('{}'.format(title))


#T-SNE Functions
#create t-distributed stochastic neighborhood embedding, a tool to visualize high dimensional data
#standard perplexiy of 800 worked the best, tried 1, 10, 100, 700, 800, 900, 1000
def tsne_model(topicspace_matrix, num_dimensions, perplexity, n_iter):
    model_tsne = TSNE(n_components=num_dimensions, perplexity=perplexity, random_state=42, n_iter=n_iter)
    tsne_matrix = model_tsne.fit_transform(topicspace_matrix)
    return model_tsne, tsne_matrix

def tsne_df_2D(matrix, topic_space, df):
    df_2d_tsne = pd.DataFrame(matrix, columns=['X','Y'])
    df_2d_tsne['Year'] = df['year']
    df_2d_tsne['Topic'] = topic_space.argmax(axis=1)
    return df_2d_tsne
def tsne_df_3D(matrix, topic_space, df):
    df_3d_tsne = pd.DataFrame(matrix, columns=['X','Y','Z'])
    df_3d_tsne['Year'] = df['year']
    df_3d_tsne['Topic'] = topic_space.argmax(axis=1)
    return df_3d_tsne

def tsne_2d_plot(tsne_df, year, topics):
    plt.style.use('fivethirtyeight')
    data = tsne_df[tsne_df['Year'] <= year]
    plt.figure(figsize=(10,10))
    colors = plt.cm.Spectral(np.linspace(0, 1, 15))
    patches = []
    for i in range(15):
        data_temp = data[data['Topic'] == i]
        color = colors[i]
        plt.scatter(data=data_temp, x='X', y='Y', s=50, c=color)
        patches.append(mpatches.Patch(color=color, label= topics[i]))
    plt.axis('off')
    plt.legend(handles=patches, ncol=5, bbox_to_anchor=(1.5, .02), fontsize=15)
    plt.title('Topic Clusters', fontsize=30)
    plt.suptitle('NIPS Papers published from 1987 to {}'.format(str(year)), fontsize=20, y=0.875)

def create_multiple_tsne_plot_2d(tsne_df, year_one, year_two, year_three, year_four, topics):
    data_one = tsne_df[tsne_df['Year'] < year_one]
    data_two = tsne_df[(tsne_df['Year'] < year_two)&(tsne_df['Year'] >= year_one)]
    data_three = tsne_df[(tsne_df['Year'] < year_three)&(tsne_df['Year'] >= year_two)]
    data_four = tsne_df[(tsne_df['Year'] < year_four)&(tsne_df['Year'] >= year_three)]
    # Create Figure
    fig, ax= plt.subplots(2, 2, figsize=(20,20))
    colors = plt.cm.Spectral(np.linspace(0, 1, 15))
    patches = []
    ax1=ax[0][0]
    ax2=ax[0][1]
    ax3=ax[1][0]
    ax4=ax[1][1]
    for i in range(15):
        data_temp = data_one[data_one['Topic'] == i]
        color = colors[i]
        ax1.scatter(data=data_temp, x='X', y='Y', s=50, c=color)
        patches.append(mpatches.Patch(color=color, label= topics[i]))
    for i in range(15):
        data_temp = data_two[data_two['Topic'] == i]
        color = colors[i]
        ax2.scatter(data=data_temp, x='X', y='Y', s=50, c=color)
    for i in range(15):
        data_temp = data_three[data_three['Topic'] == i]
        color = colors[i]
        ax3.scatter(data=data_temp, x='X', y='Y', s=50, c=color)
    for i in range(15):
        data_temp = data_four[data_four['Topic'] == i]
        color = colors[i]
        ax4.scatter(data=data_temp, x='X', y='Y', s=50, c=color)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')
    plt.legend(handles=patches, ncol=1, bbox_to_anchor=(1.8, 2.1), fontsize=20)
    plt.title('t-SNE Clusters by Topic Over the Years', fontsize=40, x=0, y=2.3)
    plt.figtext(0.1,0.88,'NIPS Papers published from 1987 to {}'.format(str(year_one)), fontsize=30)
    plt.figtext(.6,0.88,'NIPS Papers published from {} to {}'.format(str(year_one),str(year_two)), fontsize=30)
    plt.figtext(0.1,0.45,'NIPS Papers published from {} to {}'.format(str(year_two),str(year_three)), fontsize=30)
    plt.figtext(0.6,0.45,'NIPS Papers published from {} to {}'.format(str(year_three),str(year_four)), fontsize=30)


def tsne_3d_plot(tsne_df, year, topics):
    data = tsne_df[tsne_df['Year'] <= year]
    plt.figure(figsize=(10,10))
    colors = plt.cm.Spectral(np.linspace(0, 1, 15))
    patches = []
    for i in range(15):
        data_temp = data[data['Topic'] == i]
        color = colors[i]
        plt.scatter(data=data_temp, x='X', y='Y', z='Z', s=50, c=color)
        patches.append(mpatches.Patch(color=color, label= topics[i]))
    plt.axis('off')
    plt.legend(handles=patches, ncol=3, bbox_to_anchor=(1.15, 0.05), fontsize=15)
    plt.title('t-SNE Clusters by Topic', fontsize=30)
    plt.suptitle('NIPS Papers published from 1987 to {}'.format(str(year)), fontsize=20, y=0.88)

#function for finding similarity
def get_similar(topics, n=None):
    """
    calculates which papers are most similar to the papers provided. Does not return
    the papers that were provided
    """
    topics = [paper for paper in topics if paper in dists.columns]
    dists_summed = dists[topics].apply(lambda row: np.sum(row), axis=1).sort_values(ascending=False)
    ranked_dists = dists_summed.index[dists_summed.index.isin(topics)==False]
    ranked_dists = ranked_dists.tolist()
    if n is None:
        return ranked_dists
    else:
        return ranked_dists[:n]
