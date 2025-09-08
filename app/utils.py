import pdfplumber
import re
import geocoder
from geopy.geocoders import Nominatim
import nltk
from nltk.corpus import stopwords

import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    

def preprocess_resume_text(raw_text: str, min_words: int = 100) -> str:
    import re, nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize
    
    # Clean basic
    text = re.sub(r'\s+', ' ', raw_text).strip()
    text = re.sub(r"[^a-zA-Z0-9.,;:!?()\-\n ]", "", text)

    # Sentence split
    sentences = sent_tokenize(text)

    # Keep only meaningful sentences (5+ words)
    sentences = [s for s in sentences if len(s.split()) >= 5]

    # Join back
    cleaned_text = " ".join(sentences)

    # Safety: if cleaning removed too much, fall back to original
    if len(cleaned_text.split()) < min_words:
        return raw_text  
    return cleaned_text


def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else "Not detected"

def extract_phone(text):
    match = re.search(r'(\+?\d{1,4}[\s-]?)?(\(?\d{3,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{4}', text)
    return match.group(0) if match else "Not detected"

def extract_name(text):
    # Pick the first line with 2 capitalized words (very naive)
    lines = text.splitlines()
    for line in lines:
        words = line.strip().split()
        if len(words) >= 2 and all(w[0].isupper() for w in words[:2]):
            return " ".join(words[:2])
    return "Not detected"


def get_location_info():
    try:
        g = geocoder.ip('me')
        latlng = g.latlng

        if not latlng:
            return None, None, None

        geolocator = Nominatim(user_agent="resume-analyzer")
        location = geolocator.reverse(latlng, language='en')
        address = location.raw['address']

        city = address.get('city', address.get('town', address.get('village', '')))
        state = address.get('state', '')
        country = address.get('country', '')

        return city, state, country

    except Exception as e:
        print(f"[Location Error] {e}")
        return None, None, None
    


# def get_recommended_skills(role):
#     skills = {
#         "Data Scientist": ["Pandas", "Numpy", "Matplotlib"],
#         "ML Engineer": ["TensorFlow", "PyTorch", "Scikit-learn"],
#         "Web Developer": ["React", "Node.js", "MongoDB"],
#         "UI/UX Designer": ["Figma", "Adobe XD", "Prototyping"],
#         "Backend Developer": ["Django", "PostgreSQL", "REST APIs"],
#         "DevOps Engineer": ["Docker", "Kubernetes", "CI/CD"]
#     }
#     return skills.get(role, [])

def calculate_resume_score(text, summary, skills, name, email, phone):
    score = 0

    # Contact info (max 20)
    if name and name != "Not detected": score += 7
    if email and email != "Not detected": score += 7
    if phone and phone != "Not detected": score += 6

    # Summary quality (max 20)
    if summary:
        word_len = len(summary.split())
        if word_len > 30:  # at least 2–3 sentences
            score += 10
        if word_len > 50:  # longer, detailed
            score += 20
        else:
            score += 10  # partial credit

    # Skills coverage (max 20)
    if skills:
        skill_count = len(skills)
        if skill_count >= 3:
            score += 10
        if skill_count >= 5:
            score += 15
        if skill_count >= 8:
            score += 20

    # Keyword relevance (max 20)
    keywords = ["python", "tensorflow", "pytorch", "docker", "react", 
                "ci/cd", "postgresql", "kubernetes", "aws", "node.js", 
                "figma", "java", "spring", "sql"]
    matched = sum(1 for kw in keywords if kw in text.lower())
    score += min(matched * 2, 20)

    # Resume length (max 20)
    word_count = len(text.split())
    if 400 <= word_count <= 1200:
        score += 20  # ideal length
    elif 250 <= word_count < 400 or 1200 < word_count <= 2000:
        score += 10  # acceptable
    else:
        score += 5   # too short/too long but still valid

    return min(score, 90)


