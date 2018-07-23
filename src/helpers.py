import datetime
import os
import re

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import requests
from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud


def text_clean_summary(text, characters):
    """
        text: a string
        return: modified initial string
    """
    # text_ = text.lower()
    text_ = re.sub("\d", " ", text)

    for character in characters:
        text_ = text_.replace(character, " ")

    if re.search('DAS THEMA DES TAGES', text_) or re.search('DIE LAGE', text_) or re.search('SPIEGEL ONLINE', text_):
        return 1
    else:
        return 0


def text_prepare(text, stop_words, characters):
    """
        text: a string 
        return: modified initial string
    """
    STOPWORDS = list(set(stop_words))

    text_ = text.lower()
    text_ = re.sub("\d", " ", text_)

    for character in characters:
        text_ = text_.replace(character, " ")

    # initialize tokenizer and tokenize text
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    list_words = tokenizer.tokenize(text_)

    # initialize german stemmer
    stemmer = nltk.stem.snowball.GermanStemmer()

    text_ = " ".join([stemmer.stem(word) for word in list_words if word not in STOPWORDS])
    return text_


def tfidf_features(X, min_df, max_df, n_gram_min, n_gram_max):
    """
        X — samples        
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result
    tfidf_vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=(n_gram_min, n_gram_max),
                                       token_pattern="(\S+)")
    X_freq = sp_sparse.vstack([sp_sparse.csr_matrix(tfidf_vectorizer.fit_transform(X))])

    return X_freq, tfidf_vectorizer.vocabulary_


def google_translator(query, language):
    url = "https://translation.googleapis.com/language/translate/v2"
    google_key = os.environ['GOOGLE_TRANSLATE_KEY']

    params = {
        'key': google_key,
        'q': query,
        'target': language
    }
    header = {'Content-Type': "application/json; charset=utf-8"}
    resp = requests.post(url=url, params=params, headers=header)
    resp = resp.json()
    try:
        return resp['data']['translations'][0]['translatedText']
    except:
        return ''


def get_quarter_boundaries(year):
    # Define datetime boundaries per quarter for the given year
    q1 = [datetime.datetime.strptime('01.01.{}'.format(year), "%d.%m.%Y"),
          datetime.datetime.strptime('30.04.{}'.format(year), "%d.%m.%Y")]
    q2 = [datetime.datetime.strptime('01.05.{}'.format(year), "%d.%m.%Y"),
          datetime.datetime.strptime('31.08.{}'.format(year), "%d.%m.%Y")]
    q3 = [datetime.datetime.strptime('01.09.{}'.format(year), "%d.%m.%Y"),
          datetime.datetime.strptime('31.12.{}'.format(year), "%d.%m.%Y")]
    return q1, q2, q3


def assign_quarter_labels(data, quarter_bound_list, quarter_label_list):
    data['quarter'] = quarter_label_list[-1]

    for i, quarter in enumerate(quarter_bound_list):
        condition = (data["date"] >= quarter[0]) & (data["date"] <= quarter[1])
        data.loc[data[condition].index, "quarter"] = quarter_label_list[i]

    return data


def text_prepare_and_extract_freqs(text, key_words, stopwords, characters):
    # Extract for the given text the frequencies of key words for later labelling
    text_ = text_prepare(text, stopwords, characters)
    freq_words = nltk.FreqDist(text_.split(" "))
    freq_words = dict(zip(list(freq_words.keys()), list(freq_words.values())))
    freqs = {k: v for k, v in freq_words.items() if k in key_words}

    if freqs.keys():
        non_present_words = [x for x in set(key_words) - set([*freqs])]
    else:
        non_present_words = key_words

    # Calculate the relative frequencies with the total number of words
    n_words = sum(list(freq_words.values()))

    for word in non_present_words:
        freqs[word] = 0

    # build final series with index ordered to concatenate series afterwards
    freq_result = pd.Series(freqs) / n_words
    freq_result.index = [idx for idx in freq_result.index]
    freq_result = freq_result.sort_index()

    return freq_result


def articles_words_tokenization_annotation(data, key_words, stopwords, characters):
    # remove undesirable characters
    # remove stopwords
    # apply stemming
    # extract key words frequency in given text
    frequencies = []
    for i, text in data["text"].iteritems():
        f = text_prepare_and_extract_freqs(text, key_words, stopwords, characters)
        frequencies.append(f)

    df_frequencies = pd.concat(frequencies, axis=1)
    df_frequencies.columns = data["time_line"]
    return df_frequencies


def articles_words_tokenization(data, stopwords, characters):
    # remove undesirable characters
    # remove stopwords
    # apply stemming
    data["text_tokenized"] = data["text"].apply(lambda x: text_prepare(x, stopwords, characters))
    return data


def articles_word_normalization(data, min_df, max_df, n_gram_min, n_gram_max):
    # calculate normalized frequencies with TF-IDF
    X_tfidf, tfidf_vocab = tfidf_features(data["text_tokenized"], min_df, max_df, n_gram_min, n_gram_max)

    # assign labels to normalized words and articles IDs
    results = pd.DataFrame(X_tfidf.todense(), columns=tfidf_vocab).T
    results.columns = data["time_line"]
    resultsT = results.T

    print("Results have {} articles and {} normalized words".format(resultsT.shape[0], resultsT.shape[1]))
    return results, resultsT


def distribution_filtering(articles, selection, vocabulary_target, quartile):
    # Calculate the sum of the selected words frequency per each article
    selection_ = pd.DataFrame(selection.loc[vocabulary_target].sum(axis=0)).reset_index().rename(
        columns={0: "freq", "time_line": "article_id"})

    # Select only those articles whose total frequency is higher than the given quartile
    a, bins = pd.qcut(selection_["freq"], 4, retbins=True)
    quartile_value = bins[quartile]
    selection_ = selection_[selection_["freq"] >= quartile_value]

    # Complete data for those selected articles (quarter, text, etc...)
    selection_complete = articles.merge(selection_, left_on="time_line", right_on="article_id")
    return selection_complete


def all_to_topic_articles(data, vocabulary_target, vocabulary_control, quartile):
    # Calculate first metric: percentage of presence of selected words after all articles words normaliztion
    present_words_target = [x for x in data.columns if x in vocabulary_target]
    percent_present_words_target = len(present_words_target) * 100 / len(vocabulary_target)

    present_words_control = [x for x in data.columns if x in vocabulary_control]
    percent_present_words_control = len(present_words_control) * 100 / len(vocabulary_control)

    # For each of the selected words extract those articles with a frequency > than the third quartile Q3
    thresholds = {}
    for word in present_words_target:
        a, bins = pd.qcut(data[data[word] > 0.0][word], 4, retbins=True)
        quartile_threshold = bins[quartile]
        thresholds[word] = quartile_threshold

    # Build dataframe conditions for selecting articles based on the thresholds    
    conditions = {k: data[k] > v for k, v in thresholds.items()}
    conditions = pd.DataFrame(conditions)
    condition = conditions.any(axis=1)
    sel_data = data[condition]

    # Calculate second metric: sum of average frequencies for the selected words in the selected articles
    freqs_selected_words = [sel_data[word].mean() for word in present_words_target]
    total_freqs = sum(freqs_selected_words)
    print("The frequencies for words: {}\nis: {}".format(present_words_target, freqs_selected_words))
    print("The sum of the frequencies is: {}".format(total_freqs))

    # compare frequencies with those for control words
    freqs_selected_words_c = [sel_data[word].mean() for word in present_words_control]
    total_freqs_c = sum(freqs_selected_words_c)
    print("The frequencies for control words: {}\nis: {}".format(present_words_control, freqs_selected_words_c))
    print("The sum of the frequencies is: {}".format(total_freqs_c))

    return sel_data, percent_present_words_target, percent_present_words_control, total_freqs, total_freqs_c


def quantify_topic_articles_per_quarter(data_total, data_topic):
    # calculate counts of articles per quarter in total and in the topic
    articles_topic = data_total.loc[data_topic.index]
    articles_topic_count = articles_topic.loc[:, ("quarter", "text")].groupby(["quarter"],
                                                                              as_index=False).count().rename(
        columns={"text": "count_topic"})
    articles_total_count = data_total.loc[:, ("quarter", "text")].groupby(["quarter"],
                                                                          as_index=False).count().rename(
        columns={"text": "count_total"})
    combine = articles_total_count.merge(articles_topic_count, left_on="quarter", right_on="quarter")

    # normalize number of articles in the topic by the total
    combine["count_norm"] = combine["count_topic"] / combine["count_total"]

    return combine


def topic_articles_normalization(data, min_df, max_df, n_gram_min, n_gram_max):
    # calculate normalized frequencies with TF-IDF
    X_tfidf, tfidf_vocab = tfidf_features(data["text_tokenized"], min_df, max_df, n_gram_min, n_gram_max)
    # tfidf_reversed_vocab = {i: word for word, i in tfidf_vocab.items()}

    # assign labels to normalized words and articles IDs
    results = pd.DataFrame(X_tfidf.todense(), columns=tfidf_vocab).T
    results.columns = data["time_line"]
    resultsT = results.T

    print("Results have {} articles and {} normalized words".format(resultsT.shape[0], resultsT.shape[1]))
    return results, resultsT


def extract_mean_freq_words_quarter(data_freqs, data_dates, vocabulary_target, vocabulary_control):
    data_freqs_ = data_freqs.reset_index()
    data_freqs_ = data_freqs_.merge(data_dates, left_on="time_line", right_on="time_line")
    try:
        data_freqs_agg = data_freqs_.drop(["text", "date", "meta_description", "text_tokenized",
                                           "time_line", "article_id", "freq"], axis=1)
    except:
        data_freqs_agg = data_freqs_.drop(["date", "meta_description", "text_tokenized", "time_line"], axis=1)
    data_freqs_agg = data_freqs_agg.groupby(["quarter"])

    sumaries = []
    metrics = []
    for i, df in data_freqs_agg:
        metrics_quarter = {}

        quarter = df["quarter"].values[0]
        df = df.drop(["quarter"], axis=1)
        summary = pd.DataFrame(df.mean(axis=0)).reset_index().rename(columns={0: "freq", "index": "word"}).sort_values(
            by="freq",
            ascending=False)

        # Calculate third metric: percentage of presence of selected words after quarter words normaliztion
        present_words_target = [x for x in summary["word"].tolist() if x in vocabulary_target]
        percent_present_words_target = len(present_words_target) * 100 / len(vocabulary_target)

        present_words_control = [x for x in summary["word"].tolist() if x in vocabulary_control]
        percent_present_words_control = len(present_words_control) * 100 / len(vocabulary_control)

        # Calculate forth metric: sum of average frequencies for the selected words in the selected articles
        freqs_selected_words = [summary[summary["word"] == word]["freq"].values[0] for word in present_words_target]
        total_freqs = sum(freqs_selected_words)
        print("The frequencies for words: {}\nis: {}".format(present_words_target, freqs_selected_words))
        print("The sum of the frequencies is: {}".format(total_freqs))

        # compare frequencies with those for control words
        freqs_selected_words_c = [summary[summary["word"] == word]["freq"].values[0] for word in present_words_control]
        total_freqs_c = sum(freqs_selected_words_c)
        print("The frequencies for control words: {}\nis: {}".format(present_words_control, freqs_selected_words_c))
        print("The sum of the frequencies is: {}".format(total_freqs_c))

        # save metrics in dict for given quarter
        metrics_quarter["quarter"] = quarter
        metrics_quarter["target_perc"] = percent_present_words_target
        metrics_quarter["control_perc"] = percent_present_words_control
        metrics_quarter["total_freqs"] = total_freqs
        metrics_quarter["total_freqs_c"] = total_freqs_c

        summary["quarter"] = quarter
        sumaries.append(summary)

        metrics.append(metrics_quarter)

    return pd.concat(sumaries), metrics


def stemm_vs_wo_stemm_mapper_builder(text, characters, stop_words):
    text_ = text.lower()
    text_ = re.sub("\d", " ", text_)

    for character in characters:
        text_ = text_.replace(character, " ")

    # initialize tokenizer and tokenize text
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    list_words = tokenizer.tokenize(text_)

    STOPWORDS = list(set(stop_words))

    # initialize german stemmer
    stemmer = nltk.stem.snowball.GermanStemmer()

    text_stemm = [stemmer.stem(word) for word in list_words if word not in STOPWORDS]
    text_swo_temm = [word for word in list_words if word not in STOPWORDS]

    # build stemmer mapper
    stemm_mapper = pd.DataFrame.from_dict(dict(zip(text_stemm, text_swo_temm)),
                                          orient="index").reset_index().rename(columns={'index': 'stemm_word',
                                                                                        0: 'original_word'})
    return stemm_mapper


def build_reverse_stemming_mapper(data, characters, stopwords):
    # extract clean words from text stemmed and not
    mappers = []
    for i, text in data['text'].iteritems():
        df = stemm_vs_wo_stemm_mapper_builder(text, characters, stopwords)
        mappers.append(df)

    # calculate the non stemmed most common word for each stemmed one
    df_mappers = pd.concat(mappers)
    df_mappers['count'] = 1
    df_mappers = df_mappers.groupby(["stemm_word", "original_word"]).count().sort_values(by="count", ascending=False) \
        .reset_index().groupby(["stemm_word"]).first().reset_index()
    df_mappers = df_mappers.loc[:, ("stemm_word", "original_word")]
    dict_mappers = dict(zip(df_mappers["stemm_word"].tolist(), df_mappers["original_word"].tolist()))
    # add missing terms
    dict_mappers['freq'] = 'frequency'

    return dict_mappers


def get_most_freq_words_quarter(data_freqs, n_top, stemming_mapper):
    ds = {}
    for i, df in data_freqs.groupby(["quarter"]):
        quarter = df["quarter"].values[0]
        df = df.loc[:, ("word", "freq")].sort_values(by="freq", ascending=False).head(n_top)

        # apply reverse stemming to have most likely original words
        df['word'] = df['word'].apply(lambda x: stemming_mapper[x])

        d = dict(zip(df["word"].tolist(), df["freq"].tolist()))
        ds[quarter] = d
    return ds


def visualize_wordcloud(dict_freqs, quarter, title, relative_scaling=0.5, max_words=100, background_color='black'):
    plt.figure(figsize=(10, 10))
    wordcloud = WordCloud(width=900, height=500, max_words=max_words, relative_scaling=relative_scaling,
                          normalize_plurals=False, background_color=background_color).generate_from_frequencies(
        dict_freqs[quarter])
    plt.title(f"Wordcloud for quarter {title}")
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
