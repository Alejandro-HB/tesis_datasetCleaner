from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt

def plot_wordcloud(city_wordcloud, label):
    img_format = 'eps'
    plt.figure(figsize=(10, 5))
    plt.imshow(city_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(label=label)
    #plt.savefig(f'./tripAdvisor/{label.split()[0].lower()}/figs/{img_format}/{label.split()[1].lower()}.eps', format='eps')


def jaccard_similarity(set1, set2):
    intersection = set1 & set2
    union = set1 | set2
    if not union:
        return 0.0
    return len(intersection) / len(union)


# Images
#tlaxcala_df = pd.read_csv('./data/lemmatized_data/tlaxcala_lemmatized_clip_places.csv')
#puebla_df = pd.read_csv('./data/lemmatized_data/puebla_lemmatized_clip_places.csv')

# TripAdvisor
# Original text
#tlaxcala_original_df = pd.read_csv(f'./tripAdvisor/tlaxcala/tlaxcala_all_for_bertopic.csv')
#puebla_origina_df = pd.read_csv(f'./tripAdvisor/puebla/puebla_all_for_bertopic.csv')
#
#text_tlaxcala = ' '.join(tlaxcala_original_df['review'])
#text_puebla = ' '.join(puebla_origina_df['review'])
#
#wordcloud_tlaxcala_original = WordCloud(max_words=500, background_color='white').generate(text_tlaxcala)
#plot_wordcloud(wordcloud_tlaxcala_original, 'Tlaxcala Original Word Cloud')
#
#wordcloud_puebla_original = WordCloud(max_words=500, background_color='white').generate(text_puebla)
#plot_wordcloud(wordcloud_puebla_original, 'Puebla Original Word Cloud')

# Lemmatized text (Tripadvisor)
#N = 500
#tlaxcala_df = pd.read_csv('./tripAdvisor/lemmatized_data/tlaxcala_lemmatized.csv')
#puebla_df = pd.read_csv('./tripAdvisor/lemmatized_data/puebla_lemmatized.csv')


# Lemmatized text (Images)
model = 'clip_places'
N = 500
tlaxcala_df = pd.read_csv(f'./data/lemmatized_data/lemmatized_no_numbers/tlaxcala_lemmatized_{model}.csv')
puebla_df = pd.read_csv(f'./data/lemmatized_data/lemmatized_no_numbers/puebla_lemmatized_{model}.csv')

text_tlaxcala = ' '.join(tlaxcala_df['lemmatized'])
text_puebla = ' '.join(puebla_df['lemmatized'])


wordcloud_tlaxcala = WordCloud(max_words=N, background_color='white').generate(text_tlaxcala)
plot_wordcloud(wordcloud_tlaxcala, 'Tlaxcala Lemmatized Word Cloud')

wordcloud_puebla = WordCloud(max_words=N, background_color='white').generate(text_puebla)
plot_wordcloud(wordcloud_puebla, 'Puebla Lemmatized Word Cloud')

#plt.show()

words_tlaxcala = set(wordcloud_tlaxcala.words_.keys())
words_puebla = set(wordcloud_puebla.words_.keys())

# Amount of words in wordcloud
amnt_words_tlaxcala = len(wordcloud_tlaxcala.words_)
amnt_words_puebla = len(wordcloud_puebla.words_)
print(f"Words in wordcloud tlaxcala: {amnt_words_tlaxcala}")
print(f"Words in wordcloud puebla: {amnt_words_puebla}")


# Find common words
common_words = words_tlaxcala.intersection(words_puebla)

# Find exclusive words for Tlaxcala
exclusive_tlaxcala = words_tlaxcala - words_puebla
exclusive_puebla = words_puebla - words_tlaxcala

# Save exclusive words in a txt
#with open(f'./tripAdvisor/wordcloud_overlapping_text/tripadvisor/difference_analysis/{N}_exclusive_words_tlaxcala.txt', 'w') as f:
#    f.write(f"{exclusive_tlaxcala}")
#with open(f'./tripAdvisor/wordcloud_overlapping_text/tripadvisor/difference_analysis/{N}_exclusive_words_puebla.txt', 'w') as f:
#    f.write(f"{exclusive_puebla}")

# Common words
print(f"Number of common words: {len(common_words)}")
print(f"Common words: {common_words}")

# Exclusive words
#print(f"\nNumber of exclusive words of Tlaxcala: {len(exclusive_tlaxcala)}")
#print(f"Exclusive words of Tlaxcala: {exclusive_tlaxcala}")
#print(f"\nNumber of exclusive words of Puebla: {len(exclusive_puebla)}")
#print(f"Exclusive words of Puebla: {exclusive_puebla}")

# Jaccard similarity
similarity = jaccard_similarity(words_tlaxcala, words_puebla)
print(f"Jaccard similarity: {similarity}")