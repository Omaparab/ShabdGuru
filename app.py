import pickle
import collections
import re
from flask import Flask, render_template, request, jsonify
from functools import lru_cache

# --- Helper function needed by pickle to load the model ---
def nested_ddict():
    """A helper function that returns a new defaultdict(int)."""
    return collections.defaultdict(int)

# --- Initialize Flask App ---
app = Flask(__name__)

# --- 1. Load the pre-trained models for Autofill ---
try:
    with open('marathi_models.pkl', 'rb') as f:
        models = pickle.load(f)
    bigrams = models['bigrams']
    trigrams = models['trigrams']
    print("✅ Autofill models loaded successfully!")
except FileNotFoundError:
    print("⚠️ Warning: 'marathi_models.pkl' not found. Autofill will not work.")
    bigrams, trigrams = {}, {}  # Allow app to run without autofill models

# --- 2. Enhanced Transliteration Logic ---

# Comprehensive Marathi transliteration mapping
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

# Consonants with inherent 'a' sound
CONSONANTS = {
    # Velars
    'ka': 'क', 'kha': 'ख', 'ga': 'ग', 'gha': 'घ', 'nga': 'ङ',
    # Palatals
    'cha': 'च', 'chha': 'छ', 'ja': 'ज', 'jha': 'झ', 'nya': 'ञ',
    # Retroflex
    'Ta': 'ट', 'Tha': 'ठ', 'Da': 'ड', 'Dha': 'ढ', 'Na': 'ण',
    'ta': 'ट', 'tha': 'ठ', 'da': 'ड', 'dha': 'ढ', 'na': 'ण',  # Alternate
    # Dentals
    'ta': 'त', 'tha': 'थ', 'da': 'द', 'dha': 'ध', 'na': 'न',
    # Labials
    'pa': 'प', 'pha': 'फ', 'ba': 'ब', 'bha': 'भ', 'ma': 'म',
    # Semivowels
    'ya': 'य', 'ra': 'र', 'la': 'ल', 'va': 'व', 'wa': 'व',
    # Sibilants
    'sha': 'श', 'shha': 'ष', 'Sha': 'ष', 'sa': 'स', 'ha': 'ह',
    # Conjuncts
    'ksha': 'क्ष', 'tra': 'त्र', 'gya': 'ज्ञ', 'dnya': 'ज्ञ',
    'shra': 'श्र', 'kshra': 'क्ष्र',
    # Special
    'La': 'ळ', 'la': 'ळ', 'lla': 'ळ',
    'za': 'झ', 'zha': 'झ',
    'fa': 'फ',
}

# Consonants without vowel (halant form)
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

# Vowel signs (matras) to attach to consonants
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

# Common word mappings for better accuracy
WORD_MAPPINGS = {
    'mala': 'मला',
    'tula': 'तुला',
    'tyala': 'त्याला',
    'tila': 'तिला',
    'amhi': 'आम्ही',
    'tumhi': 'तुम्ही',
    'te': 'ते',
    'ti': 'ती',
    'ha': 'हा',
    'hi': 'ही',
    'he': 'हे',
    'mi': 'मी',
    'tu': 'तू',
    'to': 'तो',
    'ahe': 'आहे',
    'hota': 'होता',
    'hoti': 'होती',
    'bhook': 'भूक',
    'lagli': 'लागली',
    'lagla': 'लागला',
    'kasa': 'कसा',
    'kashi': 'कशी',
    'kase': 'कसे',
    'kay': 'काय',
    'kuthe': 'कुठे',
    'kiti': 'किती',
    'kon': 'कोण',
    'kharach': 'खरच',
    'nahi': 'नाही',
    'ho': 'हो',
    'naahi': 'नाही',
    'mhanje': 'म्हणजे',
    'mhanun': 'म्हणून',
    'pan': 'पण',
    'ani': 'आणि',
    'ya': 'या',
    'chi': 'ची',
    'cha': 'चा',
    'che': 'चे',
}

# Numbers
NUMBERS = {
    '0': '०', '1': '१', '2': '२', '3': '३', '4': '४',
    '5': '५', '6': '६', '7': '७', '8': '८', '9': '९',
}


def preprocess_text(text):
    """Normalize input text for better transliteration."""
    # Convert to lowercase for processing
    text = text.lower()
    # Replace common variations
    text = text.replace('zh', 'jh')
    text = text.replace('x', 'ksh')
    return text


def transliterate_word(word):
    """Transliterate a single word with improved accuracy."""
    # Check if it's a known word
    word_lower = word.lower()
    if word_lower in WORD_MAPPINGS:
        return WORD_MAPPINGS[word_lower]
    
    result = ""
    i = 0
    
    while i < len(word):
        matched = False
        
        # Try to match from longest to shortest (max 5 chars)
        for length in range(min(5, len(word) - i), 0, -1):
            chunk = word[i:i+length]
            
            # Skip if ends with a vowel letter but we're not at word end
            # This prevents matching 'ka' when we should match 'k' + vowel
            if length > 1 and i + length < len(word):
                next_char = word[i+length:i+length+1]
                if next_char in 'aeiou' and chunk[-1] in 'aeiou':
                    continue
            
            # Try consonant + vowel combinations first
            if chunk in CONSONANTS:
                result += CONSONANTS[chunk]
                i += length
                matched = True
                break
            
            # Try halant consonants
            elif chunk in HALANT_CONSONANTS:
                # Check if next part is a vowel matra
                remaining = word[i+length:]
                matra_matched = False
                
                for matra_len in range(min(3, len(remaining)), 0, -1):
                    matra_chunk = remaining[:matra_len]
                    if matra_chunk in MATRAS:
                        # Add consonant base (remove halant) + matra
                        result += HALANT_CONSONANTS[chunk][:-1] + MATRAS[matra_chunk]
                        i += length + matra_len
                        matra_matched = True
                        matched = True
                        break
                
                if not matra_matched:
                    # No matra follows, keep halant
                    result += HALANT_CONSONANTS[chunk]
                    i += length
                    matched = True
                break
            
            # Try standalone vowels
            elif chunk in VOWELS:
                result += VOWELS[chunk]
                i += length
                matched = True
                break
        
        if not matched:
            # Handle single character
            char = word[i]
            if char in NUMBERS:
                result += NUMBERS[char]
            elif char.isspace() or char in '.,!?;:"\'-()[]{}':
                result += char
            else:
                # Unknown character, keep as is
                result += char
            i += 1
    
    return result


@lru_cache(maxsize=1000)
def transliterate_text(eng_text):
    """
    Enhanced transliteration with word-level processing and caching.
    """
    if not eng_text:
        return ""
    
    # Preprocess
    processed = preprocess_text(eng_text)
    
    # Split into words and transliterate each
    words = processed.split()
    transliterated_words = [transliterate_word(word) for word in words]
    
    return ' '.join(transliterated_words)


# --- 3. Optimized Prediction Logic for Autofill ---
@lru_cache(maxsize=500)
def get_suggestions(full_text):
    """Get word suggestions with caching for better performance."""
    if not full_text or not trigrams:
        return []
    
    words = full_text.strip().split()
    base_suggestions = {}
    
    # Trigram suggestions (higher priority)
    if len(words) >= 2:
        context = (words[-2], words[-1])
        if context in trigrams:
            for word, count in trigrams[context].items():
                base_suggestions[word] = count * 2  # Higher weight for trigrams
    
    # Bigram suggestions
    if len(words) >= 1:
        context = words[-1]
        if context in bigrams:
            for word, count in bigrams[context].items():
                if word not in base_suggestions:
                    base_suggestions[word] = count
    
    # Sort by frequency and return top 3
    sorted_suggestions = sorted(
        base_suggestions.items(),
        key=lambda item: item[1],
        reverse=True
    )
    
    return [word for word, count in sorted_suggestions[:3]]


# --- 4. Define Web Routes ---

@app.route('/')
def home():
    """Main route for Autofill."""
    return render_template('home.html')


@app.route('/translator')
def translator():
    """Route for the Translator page."""
    return render_template('translator.html')


@app.route('/suggest', methods=['POST'])
def suggest():
    """API route for Autofill suggestions."""
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
    """API route for Transliteration."""
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
    """Handle 404 errors."""
    return render_template('home.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    app.logger.error(f"Internal error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500


# --- 5. Run the App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)