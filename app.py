import os
from flask import Flask, render_template, request, jsonify, send_file
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import pdfplumber
from docx import Document
import pandas as pd
import json
import traceback
import numpy as np
import re
from difflib import get_close_matches
from datetime import datetime
from dateutil.parser import parse
from typing import List, Tuple, Optional
from difflib import SequenceMatcher

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the sentence transformer model
print("Initializing sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model initialized successfully")

# Load skills from JSON file with error handling
try:
    with open("skills.json", "r") as f:
        SKILL_LIST = json.load(f)
    print("Skills loaded successfully from skills.json")
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Warning: Could not load skills.json: {str(e)}")
    print("Falling back to empty skill list")
    SKILL_LIST = {}

# Global degree mappings
VALID_DEGREES = {
    # Master's degrees
    "mba": "MBA",
    "master of business administration": "MBA",
    "m.sc": "M.Sc",
    "msc": "M.Sc",
    "master of science": "M.Sc",
    "m.s": "M.S.",
    "master of technology": "M.Tech",
    "m.tech": "M.Tech",
    "m.e": "M.E.",
    "master of engineering": "M.E.",
    "m.com": "M.Com",
    "master of commerce": "M.Com",
    "mca": "MCA",
    "master of computer applications": "MCA",
    
    # Bachelor's degrees
    "b.sc": "B.Sc",
    "bachelor of science": "B.Sc",
    "b.tech": "B.Tech",
    "bachelor of technology": "B.Tech",
    "b.e": "B.E.",
    "bachelor of engineering": "B.E.",
    "b.com": "B.Com",
    "bachelor of commerce": "B.Com",
    "bba": "BBA",
    "bachelor of business administration": "BBA",
    "bca": "BCA",
    "bachelor of computer applications": "BCA",
    
    # Doctorate
    "phd": "Ph.D",
    "ph.d": "Ph.D",
    "doctor of philosophy": "Ph.D",
    "doctorate": "Ph.D"
}

# Job context keywords with expanded role variations
RELEVANT_ROLE_KEYWORDS = {
    "recruiter": [
        "recruiter", "talent acquisition", "hr executive", "sourcing",
        "recruitment", "hiring", "staffing", "talent management", "recruiting"
    ],
    "software_engineer": [
        "developer", "software engineer", "programmer", "sde",
        "full stack", "frontend", "backend", "devops", "software developer",
        "application developer", "systems engineer", "web developer"
    ],
    "designer": [
        "ui designer", "ux", "graphic designer", "product design",
        "visual design", "interaction design", "user experience", "user interface",
        "creative designer", "web designer"
    ],
    "marketing": [
        "marketing", "growth", "seo", "digital marketer",
        "content marketing", "social media", "brand", "digital marketing",
        "marketing manager", "growth hacker"
    ],
    "hr": [
        "hr", "human resources", "hrbp", "people operations",
        "employee relations", "talent management", "hr manager",
        "human capital", "workforce management"
    ],
    "sales": [
        "sales executive", "account manager", "client development",
        "business development", "sales representative", "account executive",
        "sales manager", "sales director", "client relations"
    ],
    "analyst": [
        "analyst", "data analyst", "business analyst", "financial analyst",
        "research analyst", "systems analyst", "market analyst",
        "data scientist", "business intelligence"
    ],
    "manager": [
        "manager", "project manager", "product manager", "program manager",
        "team lead", "department manager", "operations manager",
        "general manager", "senior manager"
    ],
    "qa": [
        "qa", "quality assurance", "test engineer", "testing",
        "quality analyst", "test analyst", "qa engineer",
        "software tester", "test automation"
    ],
    "support": [
        "support", "customer support", "technical support", "help desk",
        "service desk", "it support", "customer service",
        "technical assistance", "support engineer"
    ]
}

def extract_text_from_pdf(file_path):
    print(f"Extracting text from PDF: {file_path}")
    with pdfplumber.open(file_path) as pdf:
        text = '\n'.join(page.extract_text() for page in pdf.pages if page.extract_text())
    return text

def extract_text_from_docx(file_path):
    print(f"Extracting text from DOCX: {file_path}")
    doc = Document(file_path)
    text = '\n'.join(paragraph.text for paragraph in doc.paragraphs)
    return text

def normalize_skill(skill):
    """Normalize a skill name to a standard format."""
    # Convert to lowercase and strip whitespace
    skill = skill.lower().strip()
    
    # Remove special characters and extra spaces
    skill = re.sub(r'[^\w\s-]', '', skill)
    skill = re.sub(r'\s+', ' ', skill)
    
    # Check for variations in the skill list
    for standard_skill, variations in SKILL_LIST.items():
        if skill in variations:
            return standard_skill
    
    return skill

def extract_skills(text):
    """Extract and normalize skills from text."""
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Initialize set to store found skills
    found_skills = set()
    
    # Look for exact matches of skill variations
    for standard_skill, variations in SKILL_LIST.items():
        for variation in variations:
            if variation in text_lower:
                found_skills.add(standard_skill)
                break
    
    # Look for skills mentioned with common prefixes/suffixes
    additional_patterns = [
        r'(?:adobe|microsoft|ms|msft|msft\.|ms\.)\s+([a-zA-Z0-9\s]+?)(?:\s+(?:suite|software|tool|tools|application|applications|program|programs))?',
        r'([a-zA-Z0-9\s]+?)\s+(?:suite|software|tool|tools|application|applications|program|programs)'
    ]
    
    for pattern in additional_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            normalized = normalize_skill(match)
            if normalized in SKILL_LIST:
                found_skills.add(normalized)
    
    return list(found_skills)

def get_similarity_score(text1, text2):
    print("Calculating similarity score...")
    # Compute embeddings
    embedding1 = model.encode([text1])[0]
    embedding2 = model.encode([text2])[0]
    
    # Calculate cosine similarity (convert to similarity percentage)
    similarity = 1 - cosine(embedding1, embedding2)
    # Convert numpy.float32 to Python float and round to 2 decimal places
    score = round(float(similarity * 100), 2)
    print(f"Similarity score calculated: {score}%")
    return score

def parse_date_range(text: str) -> Optional[Tuple[datetime, datetime]]:
    """Parse a date range string into start and end dates."""
    # Common date range patterns
    patterns = [
        r'(?P<start>\w+\s+\d{4})\s*(?:to|-|–)\s*(?P<end>\w+\s+\d{4}|present|current)',
        r'(?P<start>\d{4})\s*(?:to|-|–)\s*(?P<end>\d{4}|present|current)',
        r'(?P<start>\w+\s+\d{4})\s*-\s*(?P<end>\w+\s+\d{4}|present|current)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                start_date = parse(match.group('start'))
                end_str = match.group('end').lower()
                
                if end_str in ['present', 'current']:
                    end_date = datetime.now()
                else:
                    end_date = parse(end_str)
                
                return (start_date, end_date)
            except (ValueError, AttributeError):
                continue
    
    return None

def has_relevant_title(section: str, target_titles: List[str], strict: bool = False) -> bool:
    """Check if a section contains a relevant job title."""
    section_lower = section.lower()
    
    for title in target_titles:
        title_lower = title.lower()
        
        if strict:
            # Exact match (case-insensitive)
            if title_lower in section_lower:
                return True
        else:
            # Fuzzy matching
            words = section_lower.split()
            for i in range(len(words) - len(title_lower.split()) + 1):
                window = ' '.join(words[i:i + len(title_lower.split())])
                if SequenceMatcher(None, title_lower, window).ratio() > 0.8:
                    return True
    
    return False

def parse_month_year(date_str: str) -> Optional[datetime]:
    """Parse a month-year string into a datetime object."""
    try:
        # Handle common month formats
        month_map = {
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05', 'jun': '06',
            'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
        }
        
        # Try different date formats
        formats = [
            '%b %Y',  # Jan 2020
            '%B %Y',  # January 2020
            '%m/%Y',  # 01/2020
            '%Y'      # 2020
        ]
        
        # Try each format
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # Try with month abbreviations
        for abbr, num in month_map.items():
            if date_str.lower().startswith(abbr):
                return datetime.strptime(f"{num}/{date_str[-4:]}", '%m/%Y')
        
        return None
    except Exception:
        return None

def parse_time_span(text: str) -> Optional[float]:
    """Extract years from time span phrases."""
    patterns = [
        r'(?P<years>\d+(?:\.\d+)?)\s*(?:years?|yrs?)',
        r'(?P<years>\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)',
        r'over\s+(?P<years>\d+(?:\.\d+)?)\s*(?:years?|yrs?)',
        r'past\s+(?P<years>\d+(?:\.\d+)?)\s*(?:years?|yrs?)',
        r'since\s+\d{4}',
        r'from\s+\d{4}\s+to\s+(?:now|present|current)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            if 'years' in match.groupdict():
                return float(match.group('years'))
            else:
                # For "since X" or "from X to now" patterns
                year_match = re.search(r'\d{4}', text)
                if year_match:
                    start_year = int(year_match.group())
                    current_year = datetime.now().year
                    return current_year - start_year
    
    return None

def extract_job_titles(text: str) -> List[str]:
    """Extract job titles from text."""
    # Common job title patterns
    patterns = [
        r'(?:position|role|title|job):?\s*([A-Za-z\s]+(?:Engineer|Developer|Manager|Designer|Analyst|Specialist|Consultant|Director|Coordinator|Recruiter|Sales|Marketing|Product|Project|Business|Data|Software|Systems|Network|Security|DevOps|QA|Test|Support|Administrator|Architect|Lead|Head|Chief|Officer|Executive|President|Vice President|VP|CTO|CIO|CFO|CEO))',
        r'(?:worked as|served as|position of|role of)\s+([A-Za-z\s]+(?:Engineer|Developer|Manager|Designer|Analyst|Specialist|Consultant|Director|Coordinator|Recruiter|Sales|Marketing|Product|Project|Business|Data|Software|Systems|Network|Security|DevOps|QA|Test|Support|Administrator|Architect|Lead|Head|Chief|Officer|Executive|President|Vice President|VP|CTO|CIO|CFO|CEO))'
    ]
    
    titles = set()
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            title = match.group(1).strip()
            if len(title.split()) <= 5:  # Avoid very long matches
                titles.add(title)
    
    return list(titles)

def extract_keywords(text: str) -> List[str]:
    """Extract relevant keywords from job description."""
    # Common job-related terms to ignore
    stop_words = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'a', 'an'}
    
    # Extract words that might be skills or responsibilities
    words = re.findall(r'\b[A-Za-z][A-Za-z0-9]+\b', text)
    
    # Filter out stop words and short words
    keywords = [word.lower() for word in words 
                if word.lower() not in stop_words 
                and len(word) > 2]
    
    # Add any skills found in the text
    skills = extract_skills(text)
    keywords.extend(skills)
    
    return list(set(keywords))  # Remove duplicates

def has_relevant_keywords(section: str, keywords: List[str]) -> bool:
    """Check if a section contains any relevant keywords."""
    section_lower = section.lower()
    return any(keyword.lower() in section_lower for keyword in keywords)

def extract_relevant_experience_years(resume_text: str, keywords: List[str]) -> Tuple[float, List[str], Optional[str], Optional[str]]:
    """Extract total years of relevant experience from resume text based on keywords."""
    # Initialize variables
    total_years = 0.0
    matched_keywords = set()
    date_ranges = []
    
    # Split resume into sections
    sections = re.split(r'\n\s*\n', resume_text)
    
    for section in sections:
        # Try to find date ranges first
        date_patterns = [
            r'(?P<start>\w+\s+\d{4})\s*(?:to|-|–)\s*(?P<end>\w+\s+\d{4}|present|current)',
            r'(?P<start>\d{4})\s*(?:to|-|–)\s*(?P<end>\d{4}|present|current)',
            r'(?P<start>\w+\s+\d{4})\s*-\s*(?P<end>\w+\s+\d{4}|present|current)'
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, section, re.IGNORECASE)
            for match in matches:
                start_date = parse_month_year(match.group('start'))
                end_str = match.group('end').lower()
                
                if end_str in ['present', 'current']:
                    end_date = datetime.now()
                else:
                    end_date = parse_month_year(end_str)
                
                if start_date and end_date:
                    date_ranges.append((start_date, end_date, section))
        
        # Check for time span phrases if no date range found
        if not date_ranges:
            time_span = parse_time_span(section)
            if time_span:
                total_years += time_span
        
        # Check for relevant keywords
        if has_relevant_keywords(section, keywords):
            # Find which keywords matched
            section_keywords = [k for k in keywords if k.lower() in section.lower()]
            matched_keywords.update(section_keywords)
    
    # Process date ranges
    relevant_date_ranges = []
    for start_date, end_date, section in date_ranges:
        if has_relevant_keywords(section, keywords):
            years = (end_date - start_date).days / 365.25
            total_years += years
            relevant_date_ranges.append((start_date, end_date))
    
    # Find earliest start and latest end dates from relevant ranges
    earliest_start = min(relevant_date_ranges, key=lambda x: x[0])[0] if relevant_date_ranges else None
    latest_end = max(relevant_date_ranges, key=lambda x: x[1])[1] if relevant_date_ranges else None
    
    return (
        round(total_years, 1),
        list(matched_keywords),
        earliest_start.strftime('%Y-%m') if earliest_start else None,
        latest_end.strftime('%Y-%m') if latest_end else None
    )

def extract_primary_job_title(text: str) -> Optional[str]:
    """Extract the most recent job title from resume text."""
    # Common section headers
    section_headers = [
        r'professional\s+experience',
        r'work\s+experience',
        r'employment\s+history',
        r'experience',
        r'work\s+history'
    ]
    
    # Find the first experience section
    for header in section_headers:
        pattern = rf'(?i){header}.*?\n(.*?)(?=\n\n|\Z)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            section = match.group(1)
            # Look for job titles in the first 1-2 entries
            entries = re.split(r'\n\s*\n', section)[:2]
            for entry in entries:
                # Common job title patterns
                title_patterns = [
                    r'(?:position|role|title|job):?\s*([A-Za-z\s]+(?:Engineer|Developer|Manager|Designer|Analyst|Specialist|Consultant|Director|Coordinator|Recruiter|Sales|Marketing|Product|Project|Business|Data|Software|Systems|Network|Security|DevOps|QA|Test|Support|Administrator|Architect|Lead|Head|Chief|Officer|Executive|President|Vice President|VP|CTO|CIO|CFO|CEO))',
                    r'(?:worked as|served as|position of|role of)\s+([A-Za-z\s]+(?:Engineer|Developer|Manager|Designer|Analyst|Specialist|Consultant|Director|Coordinator|Recruiter|Sales|Marketing|Product|Project|Business|Data|Software|Systems|Network|Security|DevOps|QA|Test|Support|Administrator|Architect|Lead|Head|Chief|Officer|Executive|President|Vice President|VP|CTO|CIO|CFO|CEO))'
                ]
                
                for pattern in title_patterns:
                    title_match = re.search(pattern, entry, re.IGNORECASE)
                    if title_match:
                        return title_match.group(1).strip()
    
    return None

def normalize_title(title: str) -> str:
    """Normalize job title for comparison."""
    if not title:
        return ""
    # Convert to lowercase and remove punctuation
    normalized = title.lower()
    normalized = re.sub(r'[^\w\s]', '', normalized)
    # Remove common prefixes/suffixes
    normalized = re.sub(r'\b(senior|junior|lead|principal|chief|head|associate|assistant)\b', '', normalized)
    return normalized.strip()

def calculate_title_similarity(title1: str, title2: str) -> Tuple[float, str]:
    """Calculate similarity between two job titles."""
    if not title1 or not title2:
        return 0.0, "No title found"
    
    # Normalize titles
    norm1 = normalize_title(title1)
    norm2 = normalize_title(title2)
    
    if norm1 == norm2:
        return 1.0, "Exact match"
    
    # Calculate similarity using SequenceMatcher
    similarity = SequenceMatcher(None, norm1, norm2).ratio()
    
    # Calculate token overlap
    tokens1 = set(norm1.split())
    tokens2 = set(norm2.split())
    overlap = len(tokens1 & tokens2) / max(len(tokens1), len(tokens2))
    
    # Combine both metrics
    final_score = (similarity + overlap) / 2
    
    # Determine match type
    if final_score >= 0.9:
        match_type = "Near exact match"
    elif final_score >= 0.7:
        match_type = "Strong match"
    elif final_score >= 0.5:
        match_type = "Partial match"
    else:
        match_type = "Weak match"
    
    return final_score, match_type

def extract_job_title(text: str) -> Optional[str]:
    """Extract job title from text, looking for common patterns."""
    # Common patterns for job titles
    patterns = [
        r'job\s*title\s*:\s*([^\n]+)',
        r'position\s*:\s*([^\n]+)',
        r'role\s*:\s*([^\n]+)',
        r'title\s*:\s*([^\n]+)',
        r'^([^.\n]+)'  # First sentence if no pattern found
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None

def normalize_title(title: str) -> List[str]:
    """Normalize a title by removing punctuation, stopwords, and converting to lowercase."""
    if not title:
        return []
    
    # Common stopwords to remove
    stopwords = {'a', 'an', 'the', 'as', 'at', 'in', 'on', 'of', 'for', 'to', 'with', 'by'}
    
    # Remove punctuation and convert to lowercase
    title = re.sub(r'[^\w\s]', '', title.lower())
    
    # Split into words and remove stopwords
    words = [word for word in title.split() if word not in stopwords]
    
    return words

def calculate_title_match(resume_title: Optional[str], job_title: Optional[str]) -> Tuple[float, str]:
    """Calculate title match percentage and return match details."""
    if not resume_title or not job_title:
        return 0.0, "No title found"
    
    # Normalize both titles
    resume_words = normalize_title(resume_title)
    job_words = normalize_title(job_title)
    
    if not job_words:
        return 0.0, "Invalid job title"
    
    # Count matching words
    matching_words = set(resume_words) & set(job_words)
    match_count = len(matching_words)
    total_job_words = len(job_words)
    
    # Calculate match percentage
    match_percentage = (match_count / total_job_words) * 100
    
    # Create match details string
    match_details = f"Resume: {resume_title}\nJob: {job_title}\nMatch Type: {'Exact' if match_percentage == 100 else 'Partial'}"
    
    return match_percentage, match_details

def extract_latest_job_title(text: str) -> str:
    """Extract the most recent job title from resume text."""
    # Common job title patterns
    patterns = [
        r'(?:worked as|position|role|title|job title|as a|as an)\s*[:]?\s*([^.,\n]+)',
        r'([^.,\n]+)\s*(?:at|in|for|with)\s+[A-Z][a-zA-Z\s]+(?:Inc\.?|LLC|Ltd\.?|Corp\.?|Company)',
        r'([^.,\n]+)\s*(?:at|in|for|with)\s+[A-Z][a-zA-Z\s]+(?:Inc\.?|LLC|Ltd\.?|Corp\.?|Company)',
    ]
    
    # Find all dates in the text
    date_pattern = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|\d{4}'
    dates = [(m.start(), m.group()) for m in re.finditer(date_pattern, text, re.IGNORECASE)]
    
    # Find all job titles
    titles = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            title = match.group(1).strip()
            # Find the closest date to this title
            title_pos = match.start()
            closest_date = None
            min_distance = float('inf')
            
            for date_pos, date in dates:
                distance = abs(title_pos - date_pos)
                if distance < min_distance:
                    min_distance = distance
                    closest_date = date
            
            titles.append({
                'title': title,
                'date': closest_date,
                'position': title_pos
            })
    
    if not titles:
        return "Not Found"
    
    # Sort titles by date (most recent first) and then by position (first mentioned)
    titles.sort(key=lambda x: (
        -int(x['date'][-4:]) if x['date'] and x['date'][-4:].isdigit() else float('inf'),
        x['position']
    ))
    
    # Get the most recent title and normalize it
    latest_title = titles[0]['title']
    return ' '.join(word.capitalize() for word in latest_title.split())

def clean_degree_text(text: str) -> str:
    """Clean and normalize degree text."""
    if not text:
        return "Not found"
        
    # Clean the text
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    
    # Check for valid degrees
    for raw, formatted in VALID_DEGREES.items():
        if raw in text:
            # Extract specialization if present
            specialization = None
            specialization_patterns = [
                r'(?:in|of|specialization|specialized|major|majoring)\s+(?:in\s+)?([^.,\n]+)',
                r'\(([^)]+)\)'
            ]
            
            for pattern in specialization_patterns:
                match = re.search(pattern, text)
                if match:
                    specialization = match.group(1).strip()
                    break
            
            # Clean up the degree
            degree = formatted
            
            # Add specialization if found
            if specialization:
                specialization = re.sub(r'\s+', ' ', specialization).strip()
                specialization = ' '.join(word.capitalize() for word in specialization.split())
                if specialization.lower() not in degree.lower():
                    degree = f"{degree} – {specialization}"
            
            return degree
    
    return "Not found"

def extract_education(text: str) -> str:
    """Extract the highest or most relevant degree from resume text."""
    # Find education section
    education_section = None
    section_headers = [
        r'education(?:al)?\s*(?:background|qualifications?)?',
        r'academic\s+(?:background|qualifications?)',
        r'qualifications?'
    ]
    
    for header in section_headers:
        matches = re.finditer(f"(?i){header}.*?(?=\n\n|\Z)", text, re.DOTALL)
        for match in matches:
            section = match.group(0)
            if len(section.split()) > 3:  # Avoid false positives
                education_section = section
                break
        if education_section:
            break
    
    # Search in education section first, then full text
    search_texts = [education_section, text] if education_section else [text]
    
    # Degree priorities (higher number = higher priority)
    degree_priorities = {
        "Ph.D": 5,
        "MBA": 4,
        "M.Tech": 4,
        "M.Sc": 4,
        "M.S.": 4,
        "M.E.": 4,
        "M.Com": 4,
        "MCA": 4,
        "B.Tech": 3,
        "B.E.": 3,
        "B.Sc": 3,
        "B.Com": 3,
        "BBA": 3,
        "BCA": 3
    }
    
    highest_degree = None
    highest_priority = -1
    
    for search_text in search_texts:
        if not search_text:
            continue
            
        # Split into lines and process each line
        lines = search_text.split('\n')
        for line in lines:
            # Skip long lines that might be titles or descriptions
            if len(line.strip()) > 60:
                continue
                
            # Clean and normalize the line
            line = re.sub(r'\s+', ' ', line).strip()
            
            # Check for valid degrees
            for raw, formatted in VALID_DEGREES.items():
                if raw in line.lower():
                    # Get the degree with specialization
                    degree = clean_degree_text(line)
                    
                    # Skip if not a valid degree
                    if degree == "Not found":
                        continue
                    
                    # Get priority
                    priority = degree_priorities.get(formatted, 0)
                    
                    # Update if higher priority found
                    if priority > highest_priority:
                        highest_priority = priority
                        highest_degree = degree
                    
                    # If same priority, prefer the one with specialization
                    elif priority == highest_priority and '–' in degree and '–' not in highest_degree:
                        highest_degree = degree
    
    if not highest_degree:
        return "Not found"
    
    return highest_degree

def extract_latest_title(text: str) -> str:
    """Extract the most recent job title from resume text."""
    # Common job title patterns with validation
    patterns = [
        r'(?:worked as|position|role|title|job title|as a|as an)\s*[:]?\s*([^.,\n]+)',
        r'([^.,\n]+)\s*(?:at|in|for|with)\s+[A-Z][a-zA-Z\s]+(?:Inc\.?|LLC|Ltd\.?|Corp\.?|Company)',
        r'([^.,\n]+)\s*(?:at|in|for|with)\s+[A-Z][a-zA-Z\s]+(?:Inc\.?|LLC|Ltd\.?|Corp\.?|Company)',
    ]
    
    # Find all dates in the text
    date_pattern = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|\d{4}'
    dates = [(m.start(), m.group()) for m in re.finditer(date_pattern, text, re.IGNORECASE)]
    
    # Find all job titles
    titles = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            title = match.group(1).strip()
            
            # Skip if title is too short or doesn't contain valid job title keywords
            if len(title) < 3 or not any(keyword in title.lower() for keyword in [
                'engineer', 'developer', 'manager', 'designer', 'analyst', 'specialist',
                'consultant', 'director', 'coordinator', 'recruiter', 'sales', 'marketing',
                'product', 'project', 'business', 'data', 'software', 'systems', 'network',
                'security', 'devops', 'qa', 'test', 'support', 'administrator', 'architect',
                'lead', 'head', 'chief', 'officer', 'executive', 'president', 'vp', 'cto',
                'cio', 'cfo', 'ceo'
            ]):
                continue
            
            # Find the closest date to this title
            title_pos = match.start()
            closest_date = None
            min_distance = float('inf')
            
            for date_pos, date in dates:
                distance = abs(title_pos - date_pos)
                if distance < min_distance:
                    min_distance = distance
                    closest_date = date
            
            titles.append({
                'title': title,
                'date': closest_date,
                'position': title_pos
            })
    
    if not titles:
        # Fallback: Look for capitalized lines with job title keywords
        for line in text.split('\n'):
            line = line.strip()
            if (len(line) > 5 and line[0].isupper() and 
                any(keyword in line.lower() for keyword in [
                    'engineer', 'developer', 'manager', 'designer', 'analyst', 'specialist',
                    'consultant', 'director', 'coordinator', 'recruiter', 'sales', 'marketing',
                    'product', 'project', 'business', 'data', 'software', 'systems', 'network',
                    'security', 'devops', 'qa', 'test', 'support', 'administrator', 'architect',
                    'lead', 'head', 'chief', 'officer', 'executive', 'president', 'vp', 'cto',
                    'cio', 'cfo', 'ceo'
                ])):
                return line
    
    if not titles:
        return "Not Found"
    
    # Sort titles by date (most recent first) and then by position (first mentioned)
    titles.sort(key=lambda x: (
        -int(x['date'][-4:]) if x['date'] and x['date'][-4:].isdigit() else float('inf'),
        x['position']
    ))
    
    # Get the most recent title and normalize it
    latest_title = titles[0]['title']
    return ' '.join(word.capitalize() for word in latest_title.split())

def get_job_context_category(job_description: str) -> Optional[str]:
    """Determine job domain by keywords with improved matching."""
    if not job_description:
        return None
        
    jd = job_description.lower()
    best_match = None
    best_score = 0
    
    for category, keywords in RELEVANT_ROLE_KEYWORDS.items():
        # Count exact matches
        exact_matches = sum(1 for k in keywords if f" {k} " in f" {jd} ")
        
        # Count partial matches
        partial_matches = sum(1 for k in keywords if k in jd)
        
        # Calculate score (weight exact matches more heavily)
        score = (exact_matches * 2) + partial_matches
        
        if score > best_score:
            best_score = score
            best_match = category
    
    return best_match

def extract_relevant_experience(text: str, job_context: Optional[str]) -> Tuple[float, Optional[str]]:
    """Scan roles and calculate time spent in matching job type with best match."""
    if not job_context or job_context not in RELEVANT_ROLE_KEYWORDS:
        return 0.0, None
        
    text = text.lower()
    total_months = 0
    best_match_title = None
    best_match_score = 0
    role_keywords = RELEVANT_ROLE_KEYWORDS[job_context]
    
    # Common date range patterns
    date_patterns = [
        r'(?P<title>.*?)\s*(?P<from>\d{4})[-–to ]+(?P<to>\d{4}|present)',
        r'(?P<from>\w+\s+\d{4})\s*(?:to|-|–)\s*(?P<to>\w+\s+\d{4}|present).*?(?P<title>.*?)(?=\n|$)',
        r'(?P<title>.*?)\s*(?P<from>\w+\s+\d{4})\s*(?:to|-|–)\s*(?P<to>\w+\s+\d{4}|present)'
    ]
    
    for pattern in date_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            title = match.group("title").strip()
            from_date = match.group("from")
            to_date = match.group("to")
            
            # Skip if title is too short
            if len(title) < 3:
                continue
            
            # Calculate match score for this title
            match_score = sum(1 for k in role_keywords if k in title)
            
            if match_score > 0:  # If there's any match
                try:
                    # Parse dates
                    if len(from_date) == 4:  # Just year
                        from_year = int(from_date)
                    else:  # Month year
                        from_year = int(from_date[-4:])
                    
                    if "present" in to_date.lower():
                        to_year = datetime.now().year
                    elif len(to_date) == 4:  # Just year
                        to_year = int(to_date)
                    else:  # Month year
                        to_year = int(to_date[-4:])
                    
                    # Calculate months
                    months = (to_year - from_year) * 12
                    total_months += months
                    
                    # Update best match if this is better
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match_title = title
                        
                except (ValueError, AttributeError):
                    continue
    
    return round(total_months / 12, 1), best_match_title

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Handle form submission
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        # Here you would typically send an email or store the message
        return render_template('contact.html', success=True)
    return render_template('contact.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/testimonials')
def testimonials():
    return render_template('testimonials.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        print("\n=== Starting Resume Analysis ===")
        
        if 'resumes[]' not in request.files:
            print("Error: No resume files provided")
            return jsonify({'error': 'No resume files provided'}), 400
        
        job_description = request.form.get('jobDescription', '').strip()
        if not job_description:
            print("Error: No job description provided")
            return jsonify({'error': 'No job description provided'}), 400
        
        print(f"Job description length: {len(job_description)} characters")
        print(f"Number of files received: {len(request.files.getlist('resumes[]'))}")

        # Extract job title and context
        job_title = extract_job_title(job_description)
        job_context = get_job_context_category(job_description)
        print(f"Extracted job title: {job_title}")
        print(f"Extracted job context: {job_context}")

        # Extract keywords from job description
        keywords = extract_keywords(job_description)
        print(f"Extracted keywords: {keywords}")

        # Extract and normalize skills from job description
        job_skills = extract_skills(job_description)
        print(f"Extracted skills from job description: {job_skills}")

        results = []
        files = request.files.getlist('resumes[]')
        
        for file in files:
            if file.filename == '':
                continue
                
            print(f"\nProcessing file: {file.filename}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            
            try:
                # Save the file
                print(f"Saving file to: {file_path}")
                file.save(file_path)
                
                # Extract text based on file extension
                if file.filename.lower().endswith('.pdf'):
                    text = extract_text_from_pdf(file_path)
                elif file.filename.lower().endswith('.docx'):
                    text = extract_text_from_docx(file_path)
                else:
                    print(f"Skipping unsupported file type: {file.filename}")
                    continue
                
                print(f"Text extracted, length: {len(text)} characters")
                
                # Extract latest job title and education
                latest_title = extract_latest_title(text)
                education = extract_education(text)
                
                # Calculate similarity score
                score = get_similarity_score(text, job_description)
                
                # Extract and normalize skills from resume
                resume_skills = extract_skills(text)
                print(f"Extracted skills from resume: {resume_skills}")
                
                # Extract relevant experience based on keywords
                experience, matched_keywords, earliest_start, latest_end = extract_relevant_experience_years(text, keywords)
                relevant_experience, best_match_title = extract_relevant_experience(text, job_context)
                print(f"Extracted total experience: {experience} years")
                print(f"Extracted relevant experience: {relevant_experience} years")
                print(f"Best matching title: {best_match_title}")
                print(f"Matched keywords: {matched_keywords}")
                print(f"Experience range: {earliest_start} to {latest_end}")
                
                # Find matched skills using set intersection
                matched_skills = list(set(resume_skills) & set(job_skills))
                print(f"Matched skills: {matched_skills}")
                
                # Build reason summary
                reason_parts = []
                
                # Add skills match info
                if matched_skills:
                    skills_count = len(matched_skills)
                    total_skills = len(job_skills)
                    top_skills = matched_skills[:3]
                    skills_text = f"Matched {skills_count} out of {total_skills} required skills"
                    if top_skills:
                        skills_text += f" ({', '.join(top_skills)})"
                    reason_parts.append(skills_text)
                
                # Add experience info
                if experience:
                    reason_parts.append(f"{experience} years total experience")
                if relevant_experience:
                    reason_parts.append(f"{relevant_experience} years in {job_context} roles")
                    if best_match_title:
                        reason_parts.append(f"Best match: {best_match_title}")
                
                # Add title match info
                if latest_title:
                    reason_parts.append(f"Title match: {latest_title} → {job_title}")
                
                reason_summary = " | ".join(reason_parts) if reason_parts else "No significant matches found"
                
                # Ensure score is a Python float and rounded to 2 decimal places
                results.append({
                    'filename': file.filename,
                    'score': round(float(score), 2),
                    'experience': f"{experience:.1f} yrs" if experience else "—",
                    'relevant_experience': f"{relevant_experience:.1f} yrs" if relevant_experience else "—",
                    'latest_title': latest_title,
                    'education': education,
                    'experience_details': {
                        'total_years': experience,
                        'relevant_years': relevant_experience,
                        'best_match_title': best_match_title,
                        'matched_keywords': matched_keywords,
                        'earliest_start': earliest_start,
                        'latest_end': latest_end
                    },
                    'matched_skills': matched_skills[:5],  # Limit to top 5 matched skills
                    'reason_summary': reason_summary
                })
                
                # Clean up the uploaded file
                print(f"Cleaning up file: {file_path}")
                os.remove(file_path)
                
            except Exception as e:
                print(f"Error processing {file.filename}:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print("Stack trace:")
                print(traceback.format_exc())
                continue
        
        if not results:
            print("No valid results generated")
            return jsonify({'error': 'No valid results could be generated from the provided files'}), 400
        
        # Sort results by score in descending order
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Save results to CSV with rounded values
        print("\nSaving results to CSV")
        df = pd.DataFrame(results)
        # Ensure all numeric values in the DataFrame are rounded
        df['score'] = df['score'].round(2)
        df.to_csv('results.csv', index=False)
        
        print("=== Analysis Complete ===")
        return jsonify(results)
        
    except Exception as e:
        print("\n=== Critical Error ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Stack trace:")
        print(traceback.format_exc())
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/download')
def download():
    return send_file('results.csv',
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name='resume_rankings.csv')

if __name__ == '__main__':
    app.run(debug=True) 