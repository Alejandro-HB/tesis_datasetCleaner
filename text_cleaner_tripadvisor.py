import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import spacy
from sklearn.feature_extraction.text import CountVectorizer

# Load Spanish language model
import es_core_news_sm
nlp = es_core_news_sm.load()

# Load data
city = 'puebla'
img_format = 'eps'
df = pd.read_csv(f'./tripAdvisor/{city}/{city}_all.csv')
df['description'] = df['Review']
df['description'] = df['description'].str.replace('_', ' ', regex=False)

# WordCloud for original text
text = ' '.join(df['description'].dropna())
wordcloud_original = WordCloud(width=800, height=400, background_color='white', max_words=500).generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_original, interpolation='bilinear')
plt.axis('off')
plt.title('Original Text Word Cloud')
plt.savefig(f'./tripAdvisor/{city}/figs/{img_format}/unprocessed_text.eps', format='eps')

# Lowercase and remove punctuation
df['description'] = df['description'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', '', x))
df['description'] = df['description'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
df['description'] = df['description'].apply(lambda x: re.sub(r'\d+', '', x))
df['description'] = df['description'].apply(lambda x: re.sub(r'[´`]', '', x))

# Tokenization, stopwords removal, and lemmatization with spaCy
def process_text(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]

df['lemmatized_tokens'] = df['description'].apply(lambda x: process_text(x))
df['lemmatized_text'] = df['lemmatized_tokens'].apply(lambda tokens: ' '.join(tokens))

# Save lemmatized text in csv
lemmatized_df = pd.DataFrame({'lemmatized': df['lemmatized_text']})
lemmatized_df.to_csv(f'./tripAdvisor/lemmatized_data/{city}_lemmatized.csv', index=False)
print(f'Lemmatized words saved to ./tripAdvisor/lemmatized_data/{city}_lemmatized.csv')

# WordCloud for lemmatized text
text = ' '.join(df['lemmatized_text'].dropna())
wordcloud_lemmatized = WordCloud(width=800, height=400, background_color='white', max_words=500).generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_lemmatized, interpolation='bilinear')
plt.axis('off')
plt.title('Lemmatized Word Cloud')
plt.savefig(f'./tripAdvisor/{city}/figs/{img_format}/lemmatized.eps', format='eps')

# Generate n-grams
def generate_ngrams(texts, n=2):
    vectorizer = CountVectorizer(ngram_range=(n, n))
    ngrams = vectorizer.fit_transform(texts)
    ngram_features = vectorizer.get_feature_names_out()
    ngram_counts = ngrams.sum(axis=0).tolist()[0]
    return pd.DataFrame({'ngram': ngram_features, 'count': ngram_counts}).sort_values(by='count', ascending=False)

for n in range(1, 4):
    print(f"Top 10 {n}-grams for Lemmatized Text:")
    print(generate_ngrams(df['lemmatized_text'], n=n).head(10))

plt.show()
plt.close('all')
