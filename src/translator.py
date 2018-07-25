import os
import requests
import numpy as np
import pandas as pd


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


def extract_n_articles_and_translate(data, n_pre_selected, n_articles, language):
    articles_selected = []
    for i, df in data.groupby(["quarter"]):
        # Select 30 articles from the top 75 with highest frequency on key words to ensure
        # that they target mainly the migration topic
        df_sort = df.sort_values(by="freq", ascending=False).reset_index(drop=True)
        df_sort = df_sort.loc[:n_pre_selected]
        articles_id = df_sort.article_id.tolist()

        # fix seed to have reproducible results and extract the selected articles
        np.random.seed(0)
        np.random.shuffle(articles_id)
        df_sort_sel = df_sort[df_sort["article_id"].isin(articles_id[:n_articles])]
        articles_selected.append(df_sort_sel)

    # check a couple of examples
    articles_selected_ = pd.concat(articles_selected).reset_index(drop=True)
    print(f"The selection of articles looks like this {articles_selected_.loc[:5].meta_description.tolist()}")

    # translate selected articles
    articles_selected_["text_en"] = articles_selected_["text"].apply(lambda x: google_translator(x, language))
    return articles_selected_
