#pip install sentence-transformers textstat nltk textblob vaderSentiment langdetect transformers language-tool-python flask flask_cors errant
#python -m spacy download en_core_web_sm

from flask import Flask, request, render_template
import language_tool_python
from gramformer import Gramformer
from sentence_transformers import SentenceTransformer, util
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import textstat
import os
from dotenv import load_dotenv
import re

app = Flask(__name__)
load_dotenv()
API_KEY=os.getenv("HF_TOKEN")

# Initialize tools and models
spell_checker = language_tool_python.LanguageTool('en-US')  # Spell-Checker
gf = Gramformer(models=1)  # Set models=1 for grammar correction
relevancy_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # Relevancy Model
semantic_model = SentenceTransformer('all-MiniLM-L6-v2',token=API_KEY)  # Semantic similarity
sentiment_analyzer = SentimentIntensityAnalyzer()  # Sentiment analysis
politeness_analyzer = SentimentIntensityAnalyzer()  # Politeness Analyzer
coherency_analyzer = SentenceTransformer('all-MiniLM-L6-v2',token=API_KEY)  # Coherency Analyzer

# Helper Functions (as you had them before)
def presentation_skill_ass(content, topic):
    # Assessment Criteria
    def evaluate_spell_with_textblob(content):
        text = spell_checker.check(content)
        corrected_content = language_tool_python.utils.correct(content, text)
        lines = content.strip().split("\n")
        incorrect = 0
        for line in lines:
            if line != "":
                prev_words = line.split(" ")
                incorrect_line = spell_checker.check(line)
                correct_line = language_tool_python.utils.correct(line, incorrect_line)
                new_words = correct_line.split(" ")
                for prev_word, new_word in zip(prev_words, new_words):
                    if prev_word != new_word:
                        incorrect += 1
        return incorrect, corrected_content

    def grammatical_error(content):
        sp_lines = content.strip().split("\n")
        lines = []

        for line in sp_lines:
            lis = line.split(".")
            for l in lis:
                if l != "":
                    lines.append(l.strip())

        incorrect = 0
        errors = {}

        for line in lines:
            if line != "":
                line = re.sub(r"[^\w\s\']", "", line)
                prev_words = line.split(" ")

                corrected_line = gf.correct(line)
                corrected_line_without_pun = next(iter(corrected_line))
                corrected_line_without_pun = re.sub(r"[^\w\s\']", "", corrected_line_without_pun)
                new_words = corrected_line_without_pun.split()

                for prev_word, new_word in zip(prev_words, new_words):
                    if prev_word != new_word:
                        errors[line] = corrected_line_without_pun
                        incorrect += 1
                        break

        return incorrect, round((10 - ((incorrect * 10) / len(lines))), 0), errors

    def evaluate_relevance(content, topic):
        topic_embedding = relevancy_model.encode(topic, convert_to_tensor=True)
        speech_embedding = relevancy_model.encode(content, convert_to_tensor=True)

        # Compute similarity
        similarity_score = util.cos_sim(topic_embedding, speech_embedding).item()

        if similarity_score < 0.45:
            similarity_tag = "Irrelevant"
        elif similarity_score < 0.5:
            similarity_tag = "Relevant"
        else:
            similarity_tag = "Highly Relevant"

        return round(similarity_score, 2), similarity_tag

    def evaluate_sentiment(content):
        sentences = sent_tokenize(content)
        scores = [sentiment_analyzer.polarity_scores(sent)['compound'] for sent in sentences]
        avg_sentiment = sum(scores) / len(scores) if scores else 0

        if avg_sentiment > 0:
            sentiment_tag = "Positive"
        elif avg_sentiment < 0:
            sentiment_tag = "Negative"
        else:
            sentiment_tag = "Neutral"
        return round(avg_sentiment, 2), sentiment_tag

    def evaluate_readability(content):
        flesch_score = textstat.flesch_reading_ease(content)

        if flesch_score > 60:
            flesch_tag = "Easy to Read"
        elif 30 < flesch_score < 60:
            flesch_tag = "Moderate Readability"
        else:
            flesch_tag = "Difficult to Read"

        return round(flesch_score, 2), flesch_tag

    def detect_politeness(text):
        politeness_score = 0

        polite_phrases = ["please", "kindly", "thank you", "would you", "could you", "I would appreciate", "sorry", "I apologize"]
        politeness_score += sum(phrase in text.lower() for phrase in polite_phrases)

        modal_verbs = ["would", "could", "may", "might", "should"]
        politeness_score += sum(modal in text.lower() for modal in modal_verbs)

        indirect_phrases = ["I was wondering", "is it possible", "could you please", "would it be okay if"]
        politeness_score += 2 * sum(phrase in text.lower() for phrase in indirect_phrases)

        greetings = ["dear", "hello", "good morning", "good evening"]
        closings = ["sincerely", "kind regards", "best regards", "yours faithfully"]
        politeness_score += sum(phrase in text.lower() for phrase in greetings + closings)

        gratitude_phrases = ["thank you", "I appreciate", "thanks", "I am grateful"]
        apology_phrases = ["I apologize", "sorry", "excuse me"]
        politeness_score += sum(phrase in text.lower() for phrase in gratitude_phrases + apology_phrases)

        hedging_phrases = ["it seems", "I think", "perhaps", "it might be better to"]
        politeness_score += sum(phrase in text.lower() for phrase in hedging_phrases)

        if text.lower().startswith("send") or "do this" in text.lower():
            politeness_score -= 1

        if len(text.split(',')) > 1:
            politeness_score += 1

        from textblob import TextBlob
        sentiment = TextBlob(text).sentiment.polarity
        if sentiment > 0:
            politeness_score += 1

        if politeness_score > 6:
            return politeness_score, "Highly Polite"
        elif 3 < politeness_score <= 6:
            return politeness_score, "Moderately Polite"
        else:
            return politeness_score, "Impolite or Neutral"

    def coherence(content):
        sentences = content.split(".")
        embeddings = coherency_analyzer.encode(sentences, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(embeddings, embeddings)

        score = 0
        for i in range(len(similarities)):
            for j in range(len(similarities[0])):
                if j != i:
                    score += similarities[i][j]
        n = len(sentences)
        coherence_score = score / ((n**2) - n)
        coherence_tag = "Excellent" if coherence_score >= 0.35 else "Good" if 0.20 < coherence_score < 0.35 else "Poor"

        return round(float(coherence_score), 2), coherence_tag

    def email_structure(email):
        structure_categories = {
            "subject": ["Subject"],
            "greeting": [
                "Dear", "Hi", "Hello", "To Whom It May Concern", "Respected",
                "Good Morning", "Good Afternoon", "Good Evening", "Greetings",
                "Dear Sir", "Dear Madam", "Dear Team", "Dear All", "Hi Everyone", "Hello Team",
                "Dear Colleagues", "Dear Members"
            ],
            "closing": [
                "Best regards", "Sincerely", "Warm regards", "Thank you", "Thanking you", "Thanks",
                "Yours sincerely", "Yours faithfully", "Kind regards", "Warm wishes", "Best wishes",
                "Regards", "Thanks and regards", "Cheers", "Take care", "Respectfully",
                "Looking forward to your response", "With appreciation", "Cordially",
                "Yours truly", "Faithfully yours"
            ]
        }

        lines = email.strip().split("\n")

        results = {
            "has_subject": False,
            "has_greeting": False,
            "has_closing": False,
        }

        score = 0

        for line in lines:
            if line.startswith("Subject"):
                score = score + 1 if results["has_subject"] == False else score
                results["has_subject"] = True
                break

        for line in lines:
            for greeting in structure_categories["greeting"]:
                if greeting in line:
                    score = score + 1 if results["has_greeting"] == False else score
                    results["has_greeting"] = True
                    break

        for line in lines:
            for closing in structure_categories["closing"]:
                if closing in line:
                    score = score + 1 if results["has_closing"] == False else score
                    results["has_closing"] = True
                    break

        return score, results
    
    score=0
    
    # Perform Assessments
    spell_inc_score, corrected_content = evaluate_spell_with_textblob(content)
    grammatical_inc_score, grammatical_score_of_10, grammatical_errors = grammatical_error(content)
    score+=grammatical_score_of_10
    relevance_score, relevancy_tag = evaluate_relevance(str(corrected_content), topic)
    score=score+10 if relevancy_tag=="Highly Relevant" else score+8 if relevancy_tag=="Relevant" else score+2
    sentiment_score, sentiment_tag = evaluate_sentiment(str(corrected_content))
    score=score+10 if sentiment_tag=="Positive" else score+9
    readability_score, readability_tag = evaluate_readability(str(corrected_content))
    score=score+10 if readability_tag=="Easy to Read" else score+8 if readability_tag=="Moderate Readability" else score+4
    politeness_score, politeness_tag = detect_politeness(str(corrected_content))
    score=score+10 if politeness_tag=="Highly Polite" else score+8 if politeness_tag=="Moderately Polite" else score+3
    coherency_score, coherency_tag = coherence(str(corrected_content))
    score=score+10 if coherency_tag=="Excellent" else score+7 if coherency_tag=="Good" else score+3
    email_structure_score, email_structure_metric = email_structure(content)
    score+=email_structure_score

    #normalizing score
    score=round((score*10)/63,2)


    # Generate Feedback
    feedback = {
        "Grammar incorections": grammatical_inc_score,
        "Grammar_score": f"{grammatical_score_of_10}/10",
        # "Grammatical_errors": grammatical_errors,
        "Relevancy_type": relevancy_tag,
        "Sentiment_type": sentiment_tag,
        "Readability_type": readability_tag,
        "Politeness type": politeness_tag,
        "Coherency_type": coherency_tag,
        "Email_structure_score": f"{email_structure_score}/3",
        # "Email_structure_type": email_structure_metric,
        "Overall Score": f"{score}/10"
    }

    return feedback

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        content = request.form['content']
        topic = request.form['topic']

        if not content or not topic:
            return render_template('index.html', error="Both 'content' and 'topic' are required.")

        feedback = presentation_skill_ass(content, topic)
        return render_template('index.html', feedback=feedback)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)