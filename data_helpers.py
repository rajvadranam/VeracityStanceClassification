from __future__ import print_function

import itertools
from collections import Counter

from gensim.models import word2vec

from os.path import join, exists, split

import os
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tensorflow.contrib import learn


def train_word2vec(sentence_matrix, vocabulary_inv,

                   num_features=300, min_word_count=1, context=10, overrite=True):
    """

    Trains, saves, loads Word2Vec model

    Returns initial weights for embedding layer.



    inputs:

    sentence_matrix # int matrix: num_sentences x max_sentence_len

    vocabulary_inv  # dict {int: str}

    num_features    # Word vector dimensionality

    min_word_count  # Minimum word count

    context         # Context window size

    """
    VectorForScentence = {}
    model_dir = 'models'

    model_name = "{:d}features_{:d}minwords_{:d}context".format(num_features, min_word_count, context)

    model_name = join(model_dir, model_name)

    if exists(model_name) and not overrite:

        embedding_model = word2vec.Word2Vec.load(model_name)

        print('Load existing Word2Vec model \'%s\'' % split(model_name)[-1])

    else:

        # Set values for various parameters

        num_workers = 6  # Number of threads to run in parallel

        downsampling = 1e-3  # Downsample setting for frequent words

        # Initialize and train the model

        print('Training Word2Vec model...')

        sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]

        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers,

                                            size=num_features, min_count=min_word_count,

                                            window=context, sample=downsampling)

        wordvectors = embedding_model.wv

        VectorForWord = []
        i = 1
        for scentence in sentences:
            for word in scentence:
                VectorForWord.append(wordvectors.word_vec(word))
            VectorForScentence[str(i) + " ".join(scentence)] = np.average(np.asarray(VectorForWord), axis=1)
            i += 1
            VectorForWord = []

        # If we don't plan to train the model any further, calling

        # init_sims will make the model much more memory-efficient.

        embedding_model.init_sims(replace=True)

        # Saving the model for later use. You can load it later using Word2Vec.load()

        if not exists(model_dir):
            os.mkdir(model_dir)

        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])

        embedding_model.save(model_name)

    # add unknown words

    embedding_weights = {key: embedding_model[word] if word in embedding_model else

    np.random.uniform(-0.25, 0.25, embedding_model.vector_size)

                         for key, word in vocabulary_inv.items()}

    return embedding_weights, VectorForScentence


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def pad_sentences(sentences, padding_word="<PAD/>"):
    """

    Pads all sentences to the same length. The length is defined by the longest sentence.

    Returns padded sentences.

    """

    sequence_length = max(len(x) for x in sentences)
    if sequence_length < 200:
        sequence_length = 200

    padded_sentences = []

    for i in range(len(sentences)):
        sentence = sentences[i]

        num_padding = sequence_length - len(sentence)

        new_sentence = sentence + [padding_word] * num_padding

        padded_sentences.append(new_sentence)

    return padded_sentences


def build_vocab(sentences):
    """

    Builds a vocabulary mapping from word to index based on the sentences.

    Returns vocabulary mapping and inverse vocabulary mapping.

    """

    # Build vocabulary

    word_counts = Counter(itertools.chain(*sentences))

    # Mapping from index to word

    vocabulary_inv = [x[0] for x in word_counts.most_common()]

    # Mapping from word to index

    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary, full):
    """

    Maps sentencs and labels to vectors based on a vocabulary.

    """

    # x_old = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    max_document_length = max([len(x.split(" ")) for x in full])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(full)))

    # look at kemans
    normalize = False
    import pandas as pd
    df1 = pd.DataFrame(x)
    df1 = df1.abs()
    if normalize:
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        scaler.fit(df1.values)
        dfnorm = scaler.transform(df1.values)
        # dfnorm =  normaliz(df1.values)
        df1Norm = pd.DataFrame(dfnorm)
    else:
        df1Norm = df1
    nclusters = int(3)
    data1 = df1Norm.values.reshape(df1Norm.values.shape[1], df1Norm.values.shape[0])

    means = KMeans(n_clusters=nclusters).fit(df1Norm)
    l = means.labels_
    cout = open("democrats_kmeans_vocab_processor.csv", encoding="utf8", mode="w")
    for j, v in zip(full, l):
        cout.write("{} ,{} \n".format(j.replace(",", " "), v))
    cout.close()
    np.savetxt("democrats_vocab_processor.csv", x, delimiter=",")

    y = np.array(labels)
    np.savetxt("democrats_label_vocab_processor.csv", y, delimiter=",")

    return [x, y, vocab_processor]


def load_and_predict_labels(data_file, train_len, pattern):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(data_file, "r", encoding='utf-8').readlines())

    # Split by words
    x_text = positive_examples
    x_text = [clean_str(sent) for sent in x_text]

    x_textlist = [x.split() for x in x_text]
    # Generate labels
    positive_labels = [pattern for _ in positive_examples]
    y = np.asarray(positive_labels)

    sentences_padded = pad_sentences(x_textlist)

    vocabulary, vocabulary_inv = build_vocab(sentences_padded)

    x, y, vocabobj = build_input_data(sentences_padded, y, vocabulary, x_text)

    vocabulary_inv = {key: value for key, value in enumerate(vocabobj.vocabulary_._mapping)}

    train_len = int(len(x) * (train_len / 100))

    np.random.seed(8)

    shuffle_indices = np.random.permutation(np.arange(len(y)))

    x = x[shuffle_indices]

    y = y[shuffle_indices]

    return [x_textlist, x, y]


def load_data_and_labels(positive_data_file, negative_data_file, neutral_data_file, train_len):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "rb").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "rb").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    neutral_examples = list(open(neutral_data_file, "rb").readlines())
    neutral_examples = [s.strip() for s in neutral_examples]
    # Split by words
    x_text = positive_examples + negative_examples + neutral_examples
    x_text = [clean_str(sent.decode("utf-8") ) for sent in x_text]

    x_textlist = [x.split() for x in x_text]
    # Generate labels
    positive_labels = [[0, 0, 1] for _ in positive_examples]
    negative_labels = [[1, 0, 0] for _ in negative_examples]
    neutral_labels = [[0, 1, 0] for _ in neutral_examples]
    y = np.concatenate([positive_labels, negative_labels, neutral_labels], 0)

    sentences_padded = pad_sentences(x_textlist)

    vocabulary, vocabulary_inv = build_vocab(sentences_padded)

    x, y, vocabobj = build_input_data(sentences_padded, y, vocabulary, x_text)

    vocabulary_inv = {key: value for key, value in enumerate(vocabobj.vocabulary_._mapping)}

    train_len = int(len(x) * (train_len / 100))

    np.random.seed(8)

    shuffle_indices = np.random.permutation(np.arange(len(y)))

    x = x[shuffle_indices]

    y = y[shuffle_indices]

    x_train = x[:train_len]

    y_train = y[:train_len]

    x_test = x[train_len:]

    y_test = y[train_len:]

    return [x_text, x_train, y_train, x_test, y_test, vocabulary, vocabulary_inv, vocabobj, x_textlist]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
