# Presentation Skill Assessment Tool

## Overview
This Flask application assesses the quality of a written text based on various metrics such as grammar, relevance, sentiment, readability, politeness, coherence, and email structure. It provides feedback and an overall score for evaluating the text's quality and alignment with the given topic.

---

## Features
- **Grammar Correction and Scoring**: Detects grammatical errors and provides a corrected version of the text.
- **Relevance Evaluation**: Assesses the relevance of the content to the provided topic.
- **Sentiment Analysis**: Determines the overall sentiment (positive, neutral, or negative) of the text.
- **Readability Scoring**: Evaluates the readability of the content using the Flesch Reading Ease score.
- **Politeness Detection**: Analyzes the tone of the content to determine politeness.
- **Coherence Scoring**: Evaluates the logical flow and coherence of the text.
- **Email Structure Assessment**: Checks for key components of a professional email such as subject, greeting, and closing.
- **Overall Score**: Combines the above metrics into a normalized score out of 10.

---

## Technologies Used
- **Flask**: Web framework for Python.
- **LanguageTool**: Grammar and spelling correction.
- **Gramformer**: Grammar correction model.
- **Sentence Transformers**: Semantic similarity and coherence analysis.
- **VADER Sentiment Analyzer**: Sentiment analysis.
- **TextBlob**: Text processing and sentiment analysis.
- **TextStat**: Readability analysis.
- **SpaCy**: Natural Language Processing.

---

## Installation

### Prerequisites
- Python 3.7 or higher
- Virtual Environment (optional but recommended)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/presentation-skill-assessment.git
   cd presentation-skill-assessment
   ```

2. **Set Up Virtual Environment** (optional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download SpaCy Model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Set Up Environment Variables**:
   - Create a `.env` file in the project root directory.
   - Add your Hugging Face API token:
     ```env
     HF_TOKEN=your_huggingface_api_token
     ```

---

## Usage

### Run the Application
```bash
python app.py
```

### Access the Web Interface
Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

### Input
- **Content**: The text you want to evaluate.
- **Topic**: The topic to assess the content's relevance.

### Output
The application provides feedback on:
- Grammar errors and score
- Relevance to the topic
- Sentiment
- Readability
- Politeness
- Coherence
- Email structure (if applicable)
- Overall score

---

## File Structure
```
project/
├── app.py                # Main application file
├── templates/
│   └── index.html        # HTML template for the web interface
├── static/
│   └── style.css         # CSS for styling the web interface
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── .env                  # Environment variables (not tracked by Git)
```

---

## Contributing
If you'd like to contribute to this project:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push to your branch.
4. Submit a pull request.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [LanguageTool](https://languagetool.org/)
- [TextBlob](https://textblob.readthedocs.io/)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [SpaCy](https://spacy.io/)
- [TextStat](https://pypi.org/project/textstat/)

