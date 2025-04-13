# Resume Matcher

A fully offline web application that ranks resumes against job descriptions using semantic similarity. Built with Flask and sentence-transformers.

## Features

- Upload multiple resumes (PDF/DOCX)
- Input job description via text area
- Drag-and-drop file upload support
- Semantic similarity scoring
- Results table with color-coded match percentages
- CSV export functionality
- 100% offline operation

## Requirements

- Python 3.8+
- Virtual environment (recommended)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Use the application:
   - Paste your job description in the text area
   - Upload resumes by dragging and dropping or using the file selector
   - Click "Match Resumes" to analyze
   - View results in the table
   - Download results as CSV if needed

## Supported File Types

- PDF (*.pdf)
- Microsoft Word (*.docx)

## How it Works

1. Text Extraction:
   - PDFs are processed using `pdfplumber`
   - DOCX files are processed using `python-docx`

2. Semantic Analysis:
   - Text is embedded using `sentence-transformers` (all-MiniLM-L6-v2 model)
   - Cosine similarity is calculated between job description and resume embeddings
   - Results are ranked by similarity score

3. Results:
   - Scores are displayed as percentages
   - Color-coded for easy interpretation:
     - Green: ≥80% match
     - Yellow: ≥60% match
     - Red: <60% match
   - Results can be exported to CSV

## Notes

- The application runs completely offline
- All processing is done locally
- No data is sent to external services
- First run may take longer as it downloads the transformer model 