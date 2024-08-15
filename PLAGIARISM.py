import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

def fetch_content(url):
    """ Fetch content from a URL and return plain text. """
    try:
        print(f"Fetching content from: {url}")  # Debug print
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text from paragraphs and remove extra whitespaces
        text = ' '.join(p.get_text() for p in soup.find_all('p'))
        if not text.strip():
            print(f"No text found at {url}")
        return text
    except requests.RequestException as e:
        print(f"Error fetching content from {url}: {e}")
        return ""

def preprocess_text(text):
    """ Preprocess text by tokenizing, removing punctuation, and stopwords. """
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

def compute_similarity(text1, text2):
    """ Compute similarity between two texts using TF-IDF and Cosine Similarity. """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]

def main():
    # Hardcoded URLs to compare
    url1 = 'https://en.wikipedia.org/wiki/Mathematics'
    url2 = 'https://en.wikipedia.org/wiki/Mathematics'

    # Fetch content from URLs
    content1 = fetch_content(url1)
    content2 = fetch_content(url2)

    if not content1 or not content2:
        print("Failed to fetch content from one or both URLs.")
        return

    # Preprocess contents
    content1_processed = preprocess_text(content1)
    content2_processed = preprocess_text(content2)

    # Compute similarity
    similarity = compute_similarity(content1_processed, content2_processed)

    print(f"Cosine Similarity: {similarity:.2f}")

    # Set a threshold for plagiarism detection
    threshold = 0.7
    if similarity > threshold:
        print("The content from the URLs is likely to be plagiarized.")
    else:
        print("The content from the URLs is likely not plagiarized.")

if __name__ == "__main__":
    main()
