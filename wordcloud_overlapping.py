from wordcloud import WordCloud
import pandas as pd


# Images
#tlaxcala_df = pd.read_csv('./data/lemmatized_data/tlaxcala_lemmatized_clip_places.csv')
#puebla_df = pd.read_csv('./data/lemmatized_data/puebla_lemmatized_clip_places.csv')

# TripAdvisor
tlaxcala_df = pd.read_csv('./tripAdvisor/lemmatized_data/tlaxcala_lemmatized.csv')
puebla_df = pd.read_csv('./tripAdvisor/lemmatized_data/puebla_lemmatized.csv')

text_tlaxcala = ' '.join(tlaxcala_df['lemmatized'])
text_puebla = ' '.join(puebla_df['lemmatized'])

wordcloud_tlaxcala = WordCloud(max_words=500).generate(text_tlaxcala)
wordcloud_puebla = WordCloud(max_words=500).generate(text_puebla)

words_tlaxcala = set(wordcloud_tlaxcala.words_.keys())
words_puebla = set(wordcloud_puebla.words_.keys())

# Amount of words in wordcloud
amnt_words_tlaxcala = len(wordcloud_tlaxcala.words_)
amnt_words_puebla = len(wordcloud_puebla.words_)
print(f"Amount of words in wordcloud tlaxcala: {amnt_words_tlaxcala}")
print(f"Amount of words in wordcloud puebla: {amnt_words_puebla}")


# Find common words
common_words = words_tlaxcala.intersection(words_puebla)

print(f"Number of common words: {len(common_words)}")
print(f"Common words: {common_words}")