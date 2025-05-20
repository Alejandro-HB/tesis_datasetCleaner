import pandas as pd

def load_word_set(file_path):
    df = pd.read_csv(file_path)
    # Convert each lemmatized entry to a set of words, then flatten to one big set
    df['lemmatized'] = df['lemmatized'].astype(str).apply(lambda x: set(x.split()))
    all_words = set.union(*df['lemmatized'])
    return all_words

def jaccard_similarity(set1, set2):
    intersection = set1 & set2
    union = set1 | set2
    if not union:
        return 0.0
    print(f'Union: {union}')
    return len(intersection) / len(union)

# Load sets from both CSVs
city_1 = 'tlaxcala'
city_2 = 'puebla'
model = 'clip_places' # For Image description only

# For Tripadvisor
set1 = load_word_set(f'./tripAdvisor/lemmatized_data/{city_1}_lemmatized.csv')
set2 = load_word_set(f'./tripAdvisor/lemmatized_data/{city_2}_lemmatized.csv')

# For Image descriptions
#set1 = load_word_set(f'./data/lemmatized_data/lemmatized_no_numbers/{city_1}_lemmatized_{model}.csv')
#set2 = load_word_set(f'./data/lemmatized_data/lemmatized_no_numbers/{city_2}_lemmatized_{model}.csv')


#print(f'Unique words in Set1 ({city_1}): {len(set1)}')
#print(f'Unique words in Set2 ({city_2}): {len(set2)}')

similarity = jaccard_similarity(set1, set2)
print(f'Jaccard Similarity: {similarity:.4f}')
