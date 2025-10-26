import pickle
import collections
import re
import numpy as np
from flask import Flask, render_template, request, jsonify
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Helper function needed by pickle to load the model ---
def nested_ddict():
    """A helper function that returns a new defaultdict(int)."""
    return collections.defaultdict(int)

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Semantic Dictionary for Marathi Words ---
# This dictionary maps words to their semantic categories and related words
SEMANTIC_GROUPS = {
    # Health-related words
    'health': {
        'words': ['बरा', 'आजारी', 'बीमार', 'तब्येत', 'आरोग्य', 'दुखणे', 'वेदना', 'रोग', 'औषध'],
        'common_next': ['आहे', 'नाही', 'होतो', 'होते', 'झाला', 'झाली', 'होणे']
    },
    # Hunger/Food related
    'food': {
        'words': ['भूक', 'तहान', 'जेवण', 'खाणे', 'पिणे', 'खाल्ले', 'प्यायले'],
        'common_next': ['आहे', 'लागली', 'लागला', 'भागली', 'झाली', 'नाही']
    },
    # Emotions
    'emotion': {
        'words': ['खुश', 'दुःखी', 'राग', 'आनंद', 'चिंता', 'भीती', 'प्रेम', 'आश्चर्य'],
        'common_next': ['आहे', 'होतो', 'होते', 'झाला', 'झाली', 'वाटते']
    },
    # State/Condition
    'state': {
        'words': ['थकलो', 'थकली', 'कंटाळलो', 'कंटाळली', 'तयार', 'व्यस्त', 'मोकळा', 'मोकळी'],
        'common_next': ['आहे', 'होतो', 'होते', 'झाला', 'झाली', 'नाही']
    },
    # Weather
    'weather': {
        'words': ['गरम', 'थंड', 'पाऊस', 'उन्हाळा', 'हिवाळा', 'वारा'],
        'common_next': ['आहे', 'होता', 'होती', 'पडतो', 'पडते', 'येतो']
    }
}

# Create reverse mapping: word -> semantic groups
WORD_TO_GROUPS = {}
for group_name, group_data in SEMANTIC_GROUPS.items():
    for word in group_data['words']:
        if word not in WORD_TO_GROUPS:
            WORD_TO_GROUPS[word] = []
        WORD_TO_GROUPS[word].append(group_name)

# --- Synonym Dictionary ---
SYNONYMS = {
    'बरा': ['चांगला', 'सुखी', 'निरोगी', 'तंदुरुस्त'],
    'आजारी': ['बीमार', 'अस्वस्थ', 'रोगी'],
    'भूक': ['भूक', 'उपासमार'],
    'थकलो': ['दमलो', 'खचलो', 'श्रांत'],
    'खुश': ['आनंदी', 'प्रसन्न', 'हर्षित'],
    'दुःखी': ['खिन्न', 'उदास', 'निराश']
}

# Create reverse synonym mapping
REVERSE_SYNONYMS = {}
for key, synonyms in SYNONYMS.items():
    for synonym in synonyms:
        if synonym not in REVERSE_SYNONYMS:
            REVERSE_SYNONYMS[synonym] = []
        REVERSE_SYNONYMS[synonym].append(key)
    REVERSE_SYNONYMS[key] = synonyms

# --- 1. Load the pre-trained models for Autofill ---
try:
    with open('marathi_models.pkl', 'rb') as f:
        models = pickle.load(f)
    bigrams = models['bigrams']
    trigrams = models['trigrams']
    print("✅ Autofill models loaded successfully!")
except FileNotFoundError:
    print("⚠️ Warning: 'marathi_models.pkl' not found. Autofill will not work.")
    bigrams, trigrams = {}, {}

# --- Build vocabulary for TF-IDF ---
def build_vocabulary():
    """Build vocabulary from all available n-grams and semantic dictionary."""
    vocab = set()
    
    # Add words from bigrams
    for word in bigrams.keys():
        vocab.add(word)
        for next_word in bigrams[word].keys():
            vocab.add(next_word)
    
    # Add words from trigrams
    for word_pair in trigrams.keys():
        vocab.add(word_pair[0])
        vocab.add(word_pair[1])
        for next_word in trigrams[word_pair].keys():
            vocab.add(next_word)
    
    # Add words from semantic dictionary
    for group_data in SEMANTIC_GROUPS.values():
        vocab.update(group_data['words'])
        vocab.update(group_data['common_next'])
    
    return list(vocab)

# Initialize vocabulary
VOCABULARY = build_vocabulary() if bigrams or trigrams else []

# --- Enhanced Transliteration Logic (keeping original) ---
VOWELS = {
    'a': 'अ', 'aa': 'आ', 'A': 'आ',
    'i': 'इ', 'ii': 'ई', 'ee': 'ई', 'I': 'ई',
    'u': 'उ', 'uu': 'ऊ', 'oo': 'ऊ', 'U': 'ऊ',
    'e': 'ए', 'E': 'ए',
    'ai': 'ऐ', 'ay': 'ऐ',
    'o': 'ओ', 'O': 'ओ',
    'au': 'औ', 'aw': 'औ', 'ou': 'औ',
    'am': 'अं', 'an': 'अं', 'ang': 'अं',
    'ah': 'अः', 'aha': 'अः',
    'ri': 'ऋ', 'ru': 'ऋ',
}

CONSONANTS = {
    'ka': 'क', 'kha': 'ख', 'ga': 'ग', 'gha': 'घ', 'nga': 'ङ',
    'cha': 'च', 'chha': 'छ', 'ja': 'ज', 'jha': 'झ', 'nya': 'ञ',
    'Ta': 'ट', 'Tha': 'ठ', 'Da': 'ड', 'Dha': 'ढ', 'Na': 'ण',
    'ta': 'ट', 'tha': 'ठ', 'da': 'ड', 'dha': 'ढ', 'na': 'ण',
    'ta': 'त', 'tha': 'थ', 'da': 'द', 'dha': 'ध', 'na': 'न',
    'pa': 'प', 'pha': 'फ', 'ba': 'ब', 'bha': 'भ', 'ma': 'म',
    'ya': 'य', 'ra': 'र', 'la': 'ल', 'va': 'व', 'wa': 'व',
    'sha': 'श', 'shha': 'ष', 'Sha': 'ष', 'sa': 'स', 'ha': 'ह',
    'ksha': 'क्ष', 'tra': 'त्र', 'gya': 'ज्ञ', 'dnya': 'ज्ञ',
    'shra': 'श्र', 'kshra': 'क्ष्र',
    'La': 'ळ', 'la': 'ळ', 'lla': 'ळ',
    'za': 'झ', 'zha': 'झ',
    'fa': 'फ',
}

HALANT_CONSONANTS = {
    'k': 'क्', 'kh': 'ख्', 'g': 'ग्', 'gh': 'घ्', 'ng': 'ङ्',
    'ch': 'च्', 'chh': 'छ्', 'j': 'ज्', 'jh': 'झ्', 'ny': 'ञ्',
    'T': 'ट्', 'Th': 'ठ्', 'D': 'ड्', 'Dh': 'ढ्', 'N': 'ण्',
    't': 'त्', 'th': 'थ्', 'd': 'द्', 'dh': 'ध्', 'n': 'न्',
    'p': 'प्', 'ph': 'फ्', 'b': 'ब्', 'bh': 'भ्', 'm': 'म्',
    'y': 'य्', 'r': 'र्', 'l': 'ल्', 'v': 'व्', 'w': 'व्',
    'sh': 'श्', 'Sh': 'ष्', 's': 'स्', 'h': 'ह्',
    'L': 'ळ्', 'z': 'झ्', 'f': 'फ्',
}

MATRAS = {
    'aa': 'ा', 'A': 'ा',
    'i': 'ि', 'ii': 'ी', 'ee': 'ी', 'I': 'ी',
    'u': 'ु', 'uu': 'ू', 'oo': 'ू', 'U': 'ू',
    'e': 'े', 'E': 'े',
    'ai': 'ै', 'ay': 'ै',
    'o': 'ो', 'O': 'ो',
    'au': 'ौ', 'aw': 'ौ', 'ou': 'ौ',
    'am': 'ं', 'an': 'ं', 'ng': 'ं',
    'ah': 'ः', 'aha': 'ः',
    'ri': 'ृ', 'ru': 'ृ',
}

WORD_MAPPINGS = {
    'mala': 'मला', 'tula': 'तुला', 'tyala': 'त्याला', 'tila': 'तिला',
    'amhi': 'आम्ही', 'tumhi': 'तुम्ही', 'te': 'ते', 'ti': 'ती',
    'ha': 'हा', 'hi': 'ही', 'he': 'हे', 'mi': 'मी', 'tu': 'तू', 'to': 'तो',
    'ahe': 'आहे', 'hota': 'होता', 'hoti': 'होती',
    'bhook': 'भूक', 'lagli': 'लागली', 'lagla': 'लागला',
    'kasa': 'कसा', 'kashi': 'कशी', 'kase': 'कसे', 'kay': 'काय',
    'kuthe': 'कुठे', 'kiti': 'किती', 'kon': 'कोण',
    'kharach': 'खरच', 'nahi': 'नाही', 'ho': 'हो', 'naahi': 'नाही',
    'mhanje': 'म्हणजे', 'mhanun': 'म्हणून', 'pan': 'पण', 'ani': 'आणि',
    'ya': 'या', 'chi': 'ची', 'cha': 'चा', 'che': 'चे',
}

NUMBERS = {
    '0': '०', '1': '१', '2': '२', '3': '३', '4': '४',
    '5': '५', '6': '६', '7': '७', '8': '८', '9': '९',
}

def preprocess_text(text):
    text = text.lower()
    text = text.replace('zh', 'jh')
    text = text.replace('x', 'ksh')
    return text

def transliterate_word(word):
    word_lower = word.lower()
    if word_lower in WORD_MAPPINGS:
        return WORD_MAPPINGS[word_lower]
    
    result = ""
    i = 0
    
    while i < len(word):
        matched = False
        
        for length in range(min(5, len(word) - i), 0, -1):
            chunk = word[i:i+length]
            
            if length > 1 and i + length < len(word):
                next_char = word[i+length:i+length+1]
                if next_char in 'aeiou' and chunk[-1] in 'aeiou':
                    continue
            
            if chunk in CONSONANTS:
                result += CONSONANTS[chunk]
                i += length
                matched = True
                break
            
            elif chunk in HALANT_CONSONANTS:
                remaining = word[i+length:]
                matra_matched = False
                
                for matra_len in range(min(3, len(remaining)), 0, -1):
                    matra_chunk = remaining[:matra_len]
                    if matra_chunk in MATRAS:
                        result += HALANT_CONSONANTS[chunk][:-1] + MATRAS[matra_chunk]
                        i += length + matra_len
                        matra_matched = True
                        matched = True
                        break
                
                if not matra_matched:
                    result += HALANT_CONSONANTS[chunk]
                    i += length
                    matched = True
                break
            
            elif chunk in VOWELS:
                result += VOWELS[chunk]
                i += length
                matched = True
                break
        
        if not matched:
            char = word[i]
            if char in NUMBERS:
                result += NUMBERS[char]
            elif char.isspace() or char in '.,!?;:"\'-()[]{}':
                result += char
            else:
                result += char
            i += 1
    
    return result

@lru_cache(maxsize=1000)
def transliterate_text(eng_text):
    if not eng_text:
        return ""
    
    processed = preprocess_text(eng_text)
    words = processed.split()
    transliterated_words = [transliterate_word(word) for word in words]
    
    return ' '.join(transliterated_words)

# --- Semantic Similarity Functions ---
def get_semantic_related_words(word):
    """Get semantically related words for a given word."""
    related = set()
    
    # Check if word is in semantic groups
    if word in WORD_TO_GROUPS:
        for group_name in WORD_TO_GROUPS[word]:
            # Add all words from the same semantic group
            related.update(SEMANTIC_GROUPS[group_name]['words'])
    
    # Check synonyms
    if word in REVERSE_SYNONYMS:
        related.update(REVERSE_SYNONYMS[word])
    
    # Remove the word itself
    related.discard(word)
    
    return list(related)

def calculate_word_similarity(word1, word2):
    """Calculate similarity between two words using character n-grams."""
    if word1 == word2:
        return 1.0
    
    # Check if they're synonyms
    if word1 in REVERSE_SYNONYMS and word2 in REVERSE_SYNONYMS[word1]:
        return 0.9
    if word2 in REVERSE_SYNONYMS and word1 in REVERSE_SYNONYMS[word2]:
        return 0.9
    
    # Check if they're in the same semantic group
    if word1 in WORD_TO_GROUPS and word2 in WORD_TO_GROUPS:
        common_groups = set(WORD_TO_GROUPS[word1]) & set(WORD_TO_GROUPS[word2])
        if common_groups:
            return 0.7
    
    # Character-level similarity (Jaccard similarity on character bigrams)
    def get_char_bigrams(word):
        return set(word[i:i+2] for i in range(len(word)-1))
    
    bigrams1 = get_char_bigrams(word1)
    bigrams2 = get_char_bigrams(word2)
    
    if not bigrams1 or not bigrams2:
        return 0.0
    
    intersection = len(bigrams1 & bigrams2)
    union = len(bigrams1 | bigrams2)
    
    return intersection / union if union > 0 else 0.0

def get_semantic_predictions(last_word):
    """Get predicted next words based on semantic understanding."""
    predictions = {}
    
    # Check if the last word belongs to any semantic group
    if last_word in WORD_TO_GROUPS:
        for group_name in WORD_TO_GROUPS[last_word]:
            # Get common next words for this semantic group
            common_next = SEMANTIC_GROUPS[group_name]['common_next']
            for next_word in common_next:
                predictions[next_word] = predictions.get(next_word, 0) + 5
    
    return predictions

# --- Enhanced Prediction Logic with Semantic Similarity ---
def get_suggestions_with_similarity(full_text):
    """Get word suggestions using n-grams and semantic similarity."""
    if not full_text or not (trigrams or bigrams):
        return []
    
    words = full_text.strip().split()
    suggestions = {}
    
    # 1. Direct n-gram matches (highest priority)
    if len(words) >= 2:
        context = (words[-2], words[-1])
        if context in trigrams:
            for word, count in trigrams[context].items():
                suggestions[word] = count * 3
    
    if len(words) >= 1:
        context = words[-1]
        if context in bigrams:
            for word, count in bigrams[context].items():
                suggestions[word] = suggestions.get(word, 0) + count * 2
    
    # 2. Semantic predictions (if no direct matches or to enhance)
    if words:
        last_word = words[-1]
        semantic_preds = get_semantic_predictions(last_word)
        for word, score in semantic_preds.items():
            suggestions[word] = suggestions.get(word, 0) + score
    
    # 3. Similarity-based predictions (for unseen words)
    if words and (not suggestions or len(suggestions) < 3):
        last_word = words[-1]
        related_words = get_semantic_related_words(last_word)
        
        # Find suggestions from related words
        for related_word in related_words:
            similarity = calculate_word_similarity(last_word, related_word)
            
            # Check bigrams for related words
            if related_word in bigrams:
                for next_word, count in bigrams[related_word].items():
                    weighted_score = count * similarity * 1.5
                    suggestions[next_word] = suggestions.get(next_word, 0) + weighted_score
            
            # Check trigrams for related words
            if len(words) >= 2:
                context = (words[-2], related_word)
                if context in trigrams:
                    for next_word, count in trigrams[context].items():
                        weighted_score = count * similarity * 2
                        suggestions[next_word] = suggestions.get(next_word, 0) + weighted_score
    
    # Sort by score and return top 3
    sorted_suggestions = sorted(
        suggestions.items(),
        key=lambda item: item[1],
        reverse=True
    )
    
    return [word for word, score in sorted_suggestions[:3]]

@lru_cache(maxsize=500)
def get_suggestions(full_text):
    """Cached wrapper for suggestions."""
    return get_suggestions_with_similarity(full_text)

# --- Web Routes ---
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/translator')
def translator():
    return render_template('translator.html')

@app.route('/suggest', methods=['POST'])
def suggest():
    try:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify([])
        
        suggestions = get_suggestions(text)
        return jsonify(suggestions)
    
    except Exception as e:
        app.logger.error(f"Error in suggest endpoint: {str(e)}")
        return jsonify([]), 500

@app.route('/transliterate', methods=['POST'])
def transliterate():
    try:
        data = request.json
        eng_text = data.get('text', '').strip()
        
        if not eng_text:
            return jsonify({'marathi_text': ''})
        
        marathi_text = transliterate_text(eng_text)
        return jsonify({'marathi_text': marathi_text})
    
    except Exception as e:
        app.logger.error(f"Error in transliterate endpoint: {str(e)}")
        return jsonify({'marathi_text': '', 'error': 'Translation failed'}), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('home.html'), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Internal error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)