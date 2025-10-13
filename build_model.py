import re
import collections
import pickle

def nested_ddict():
    """A helper function that returns a new defaultdict(int)."""
    return collections.defaultdict(int)

def clean_and_tokenize(filepath):
    """Reads the corpus, cleans it, and tokenizes it into words."""
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Remove all characters that are not Devanagari or basic punctuation
    text = re.sub(r'[^\u0900-\u097F\s।?]', '', text)
    
    # --- CHANGE: Smarter Punctuation Handling ---
    # Add a space before punctuation marks so they are treated as separate tokens.
    # For example, "आहे।" becomes "आहे ।"
    text = re.sub(r'([।?])', r' \1', text)
    
    # Replace newline characters with spaces
    text = text.replace('\n', ' ')
    
    # Tokenize by splitting on spaces
    tokens = text.split()
    
    print(f"Corpus cleaned. Total tokens: {len(tokens)}")
    return tokens

def build_ngram_models(tokens):
    """Builds unigram, bigram, and trigram models from a list of tokens."""
    
    unigrams = collections.Counter(tokens)
    
    bigrams = collections.defaultdict(nested_ddict)
    for i in range(len(tokens) - 1):
        bigrams[tokens[i]][tokens[i+1]] += 1
        
    trigrams = collections.defaultdict(nested_ddict)
    for i in range(len(tokens) - 2):
        context = (tokens[i], tokens[i+1])
        trigrams[context][tokens[i+2]] += 1
        
    print("N-gram models built successfully.")
    return unigrams, bigrams, trigrams

if __name__ == "__main__":
    corpus_filepath = 'corpus.txt'
    
    word_tokens = clean_and_tokenize(corpus_filepath)
    
    unigram_model, bigram_model, trigram_model = build_ngram_models(word_tokens)
    
    with open('marathi_models.pkl', 'wb') as f:
        pickle.dump({
            'unigrams': unigram_model,
            'bigrams': bigram_model,
            'trigrams': trigram_model
        }, f)
        
    print("Models have been saved to marathi_models.pkl")