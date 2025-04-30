import pandas as pd

def load_cardinality(file_path):
    df = pd.read_csv(file_path)
    df['lemmatized'] = df['lemmatized'].apply(lambda x: set(x.split()))
    return len(set.union(*df['lemmatized']))

def jaccard_cardinality_similarity(card1, card2):
    return min(card1, card2) / max(card1, card2)



# Load cardinalities from both CSVs
city_1 = 'tlaxcala'
city_2 = 'puebla'
#model_used = 'clip_places'


cardinality1 = load_cardinality(f'./tripAdvisor/lemmatized_data/{city_1}_lemmatized.csv')
cardinality2 = load_cardinality(f'./tripAdvisor/lemmatized_data/{city_2}_lemmatized.csv')

print(f'Cardinality of Set1 ({city_1}): {cardinality1}')
print(f'Cardinality of Set2 ({city_2}): {cardinality2}')

similarity = jaccard_cardinality_similarity(cardinality1, cardinality2)
print(f'Jaccard Similarity Based on Cardinality: {similarity:.4f}')