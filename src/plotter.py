import matplotlib.pyplot as plt
from wordcloud import WordCloud


def visualize_wordcloud(dict_freqs, quarter, title, relative_scaling=0.5, max_words=100, background_color='black'):
    plt.figure(figsize=(10, 10))
    wordcloud = WordCloud(width=900, height=500, max_words=max_words, relative_scaling=relative_scaling,
                          normalize_plurals=False, background_color=background_color).generate_from_frequencies(
        dict_freqs[quarter])
    plt.title(f"Wordcloud for quarter {title}")
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
