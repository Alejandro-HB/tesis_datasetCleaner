# Script for text cleaning using NLP techniques
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from mpl_toolkits.mplot3d import Axes3D

# Generate n-grams function
def generate_ngrams(texts, n=2):
    vectorizer = CountVectorizer(ngram_range=(n, n))
    ngrams = vectorizer.fit_transform(texts)
    ngram_features = vectorizer.get_feature_names_out()
    ngram_counts = ngrams.sum(axis=0).tolist()[0]
    ngram_df = pd.DataFrame({'ngram': ngram_features, 'count': ngram_counts})
    return ngram_df.sort_values(by='count', ascending=False)

# Check and download necessary NLTK data
from nltk.data import find
try:
    find('tokenizers/punkt')
    find('corpora/stopwords')
    find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load data
city = 'tlaxcala'
folder = 'places'
img_format = 'eps'
df = pd.read_csv('./data/'+city+'.csv')
# Clip_places
#df['description'] = df['prompt'] + df['places2_description']
# Clip
#df['description'] = df['prompt']
# Places
df['description'] = df['places2_description']
# Separate words joined with underscores
df['description'] = df['description'].str.replace('_', ' ')

# Create original text word cloud
text = ' '.join(df['description'])
wordcloud_original = WordCloud(width=800, height=400, background_color='white', max_words=10000).generate(text)

# Plot it
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_original, interpolation='bilinear')
plt.axis('off')
plt.title('Original Text Word Cloud')
plt.savefig('./figs/'+city+'/'+folder+'/'+img_format+'/unprocessed_text.eps', format='eps')

# Clean up data
df.drop(columns=['prompt', 'places2_description'], inplace=True, errors='ignore')
df.drop(columns=[df.columns[0]], inplace=True, errors='ignore')
df['description'] = df['description'].str.lower()
df['description'] = df['description'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

# Create processed text word cloud
text = ' '.join(df['description'])
wordcloud_processed = WordCloud(width=800, height=400, background_color='white', max_words=10000).generate(text)

# Plot it
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_processed, interpolation='bilinear')
plt.axis('off')
plt.title('Lower and Cleaned Word Cloud')
plt.savefig('./figs/'+city+'/'+folder+'/'+img_format+'/lower_and_cleaned.eps', format='eps')

# Tokenization
df['tokenized_sentences'] = df['description'].apply(lambda x: nltk.word_tokenize(x))

# Remove stop words
stop_words = set(stopwords.words('english'))
df['tokenized_sentences'] = df['tokenized_sentences'].apply(lambda x: [word for word in x if word not in stop_words])

# Create tokenized text word cloud
text = ' '.join([' '.join(tokens) for tokens in df['tokenized_sentences']])
wordcloud_tokenized = WordCloud(width=800, height=400, background_color='white', max_words=10000).generate(text)

# Plot it
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_tokenized, interpolation='bilinear')
plt.axis('off')
plt.title('Tokenized Word Cloud')
plt.savefig('./figs/'+city+'/'+folder+'/'+img_format+'/tokenized.eps', format='eps')

# Stemming and Lemmatization
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
df['stemmed'] = df['tokenized_sentences'].apply(lambda x: [stemmer.stem(word) for word in x])
df['lemmatized'] = df['tokenized_sentences'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

#Save lemmatized text in csv
lemmatized_df = pd.DataFrame({'lemmatized': df['lemmatized'].apply(lambda x: ' '.join(x))})
lemmatized_df.to_csv(f'./data/lemmatized_data/{city}_lemmatized_{folder}.csv', index=False)
print(f"Lemmatized words saved to ./data/lemmatized_data/{city}_lemmatized_{folder}.csv")


# Plot Stemmed WordCloud
text = ' '.join([' '.join(stemmed) for stemmed in df['stemmed']])
wordcloud_stemmed = WordCloud(width=800, height=400, background_color='white', max_words=10000).generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_stemmed, interpolation='bilinear')
plt.axis('off')
plt.title('Stemmed Word Cloud')
plt.savefig('./figs/'+city+'/'+folder+'/'+img_format+'/stemmed.eps', format='eps')

# Plot Lemmatized WordCloud
text = ' '.join([' '.join(lemmatized) for lemmatized in df['lemmatized']])
wordcloud_lemmatized = WordCloud(width=800, height=400, background_color='white', max_words=10000).generate(text)
#Word cardinality
#word_freq = wordcloud_lemmatized.words_
word_freq = WordCloud().process_text(text)
print(word_freq)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_lemmatized, interpolation='bilinear')
plt.axis('off')
plt.title('Lemmatized Word Cloud')
plt.savefig('./figs/'+city+'/'+folder+'/'+img_format+'/lemmatized.eps', format='eps')

# Convert lists back to sentences for ngram analysis
df['description_original'] = df['description'].apply(lambda x: ''.join(x))
df['description_lemmatized'] = df['lemmatized'].apply(lambda x: ' '.join(x))


# Generate and display n-grams for original and lemmatized text
for n in range(1, 4):
    print(f"Top 10 {n}-grams for Original Text:")
    print(generate_ngrams(df['description_original'], n=n).head(10))

    print(f"Top 10 {n}-grams for Lemmatized Text:")
    print(generate_ngrams(df['description_lemmatized'], n=n).head(10))

plt.show()
plt.close('all')
