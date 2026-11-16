# RecruitAI - AI-Powered Recruitment Automation Platform

**RecruitAI** is an intelligent end-to-end recruitment automation platform designed to transform the talent acquisition lifecycle through AI-driven workflows. 

---

# Problem Statement

Traditional recruitment processes suffer from inefficiencies including manual resume screening, subjective candidate evaluation, and delayed decision-making. Existing Applicant Tracking Systems (ATS) rely on keyword-based matching that fails to capture semantic context, leading to false positives and missed qualified candidates.

This project aims to develop an **AI-powered recruitment platform** that automates the hiring lifecycle with objective skill validation, semantic matching, and meaningful feedback for all candidates.

---

# Key Features
Semantic Resume Parsing – Transformer-based NLP models for contextual resume understanding, skill extraction, and qualification analysis beyond traditional keyword matching.

ML-Based Candidate Fit Prediction – Combines semantic matching scores, skill verification, and qualification metrics to predict candidate-job suitability.

AI-Powered Candidate Screening & Selection – Automatically ranks candidates, identifies suitable applicants, and assists recruiters in shortlisting candidates based on job requirements.

AI-Driven Interview Question Generation – Generates personalized technical and HR interview questions based on candidate resumes, job descriptions, and identified skill sets.

Automated Interview Assessment – Evaluates candidate responses using predefined assessment criteria, identifies strengths and knowledge gaps, and generates structured interview evaluation reports.

Interview Scheduling & Calendar Integration – Supports automated interview scheduling, candidate self-scheduling, calendar integration, and interview notifications.

Personalized Learning Path Generator – Provides customized upskilling recommendations and learning resources based on identified skill gaps and assessment results.

Advanced Analytics Dashboard – Visualizes recruitment metrics, candidate rankings, selection statistics, interview performance, and skill gap analysis.

Explainable & Bias-Aware AI – Provides transparent candidate scoring, interpretable evaluation results, and bias-aware screening mechanisms to support fairer recruitment decisions.

Automated Email Notifications – Sends personalized emails for shortlisting, rejection, interview invitations, assessment requests, and learning recommendations.

---

# System Architecture

The system consists of three main layers:

### 1. Frontend Layer (HTML,CSS,JS)
Recruiter Dashboard – Job description creation, resume management, candidate ranking, shortlisting, and recruitment workflow management.

AI Interview Interface – Displays personalized interview questions and collects candidate responses for evaluation.

Analytics Dashboard – Interactive visualizations for candidate rankings, hiring metrics, interview performance, and skill gap analysis.

### 2. Backend Layer (Flask)
RESTful API Endpoints – Handles communication between frontend components and backend modules.

Resume Processing Module – Handles PDF, DOCX, and TXT resume uploads and text extraction.

Recruitment Workflow Management – Coordinates candidate screening, shortlisting, rejection, interview requests, and evaluation processes.

Interview Management Module – Manages question generation requests, candidate responses, interview assessments, and evaluation reports.

Email Automation – Sends recruitment notifications, interview invitations, assessment requests, and personalized feedback.

Analytics Module – Processes recruitment data and generates insights for the recruiter dashboard.

### 3. AI Processing Layer
Semantic Resume Understanding – Uses Transformer-based sentence embeddings and NLP techniques for contextual resume parsing and job-description matching.

Candidate Ranking Engine – Evaluates candidates using semantic similarity, skill matching, and qualification-based scoring.

ML-Based Fit Prediction – Uses machine learning models, including XGBoost and SVM, for candidate-fit classification.

AI Interview Question Generator – Generates personalized technical and HR interview questions using LLMs based on job requirements and candidate profiles.

Automated Interview Evaluation – Analyzes candidate responses against predefined evaluation criteria and generates structured assessment reports.

Explainable AI Module – Provides interpretable candidate scoring and feature-level insights using SHAP.

Learning Path Generator – Identifies skill gaps and generates personalized learning and upskilling recommendations.

---

# Technologies Used

- **Backend**: Python, Flask, Flask-CORS
- **NLP / Embeddings**: Sentence Transformers (all-MiniLM-L6-v2), spaCy, PyTorch
- **Machine Learning**: scikit-learn, XGBoost, SVM ensemble, SHAP (explainability)
- **Resume Parsing**: PyPDF2, python-docx, docx2txt
- **LLM Integration**: Groq API (LLaMA-3)
- **Automation**: APScheduler, SMTP email
- **Frontend**: HTML, CSS, JavaScript, Bootstrap, Chart.js


---
