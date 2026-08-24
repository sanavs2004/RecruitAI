import os
import re
import uuid
import json
import logging
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer, util
from config import Config

from jd_module import get_all_jds, STORE_FOLDER as JD_STORE
from modules.learning_path_emailer import send_learning_path_email
from modules.semantic_parser import SemanticParser
from modules.semantic_ranker import SemanticRanker
from modules.skill_extractor import SkillExtractor

# ── Environment setup ─────────────────────────────────────────────────────────
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ── Proper logging instead of bare print ─────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ── Comprehensive skill vocabulary for robust extraction ──────────────────────
SKILL_VOCABULARY = {
    # Programming languages
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
    'swift', 'kotlin', 'ruby', 'php', 'scala', 'r', 'matlab', 'perl',
    # Web frameworks
    'react', 'angular', 'vue', 'django', 'flask', 'fastapi', 'spring',
    'express', 'node.js', 'next.js', 'laravel', 'rails', 'asp.net',
    # Databases
    'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
    'cassandra', 'oracle', 'sqlite', 'dynamodb', 'firebase',
    # Cloud & DevOps
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform',
    'ansible', 'ci/cd', 'linux', 'git', 'github', 'gitlab',
    # ML & Data
    'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras',
    'scikit-learn', 'pandas', 'numpy', 'data science', 'nlp', 'computer vision',
    'transformers', 'bert', 'llm', 'xgboost', 'spark', 'hadoop',
    # Mobile
    'android', 'ios', 'react native', 'flutter',
    # Other
    'html', 'css', 'rest api', 'graphql', 'microservices', 'agile', 'scrum',
    'jira', 'figma', 'photoshop', 'tableau', 'power bi', 'excel',
}


class ResumeScreeningEngine:
    """
    Core resume screening engine.
    Improvements:
      1. Proper logging (no bare except/pass)
      2. Robust skill extraction using vocabulary + embeddings
      3. Batch encoding for speed
      4. SHAP integration with human-readable explanations
      5. Hybrid scoring: semantic + ML
    """

    def __init__(self, config):
        self.config = config
        logger.info("🧠 Initialising ResumeScreeningEngine...")

        # ── Core NLP modules ──────────────────────────────────────────────
        self.parser          = SemanticParser()
        self.ranker          = SemanticRanker()
        self.skill_extractor = SkillExtractor()

        # ── Single shared embedder (FIX: was loading twice before) ───────
        import torch
        torch.set_num_threads(2)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        logger.info("✅ Sentence Transformer loaded (single shared instance)")

        # ── ML Fit Predictor (XGBoost + SVM + SHAP) ───────────────────────
        try:
            from modules.fit_predictor import FitPredictor
            self.fit_predictor = FitPredictor(os.path.join(config.BASE_DIR, 'models'))
            logger.info("✅ FitPredictor loaded")
        except Exception as e:
            self.fit_predictor = None
            logger.warning(f"FitPredictor not loaded: {e}")

        # ── GitHub verification ────────────────────────────────────────────
        try:
            from modules.github_verification import GitHubVerifier
            from modules.skill_cross_verifier import SkillCrossVerifier
            self.github_verifier = GitHubVerifier(config)
            self.skill_verifier  = SkillCrossVerifier()
            self.github_enabled  = True
            logger.info("✅ GitHub verification loaded")
        except Exception as e:
            self.github_verifier = None
            self.skill_verifier  = None
            self.github_enabled  = False
            logger.warning(f"GitHub modules not loaded: {e}")

        # ── LLM (Groq API) ────────────────────────────────────────────────
        try:
            from groq import Groq
            self.client = Groq(api_key=Config.GROQ_API_KEY)
            self.llm_model     = "llama-3.3-70b-versatile"
            self.llm_available = True
            logger.info("✅ LLM (Groq API) available")
        except Exception as e:
            self.llm_available = False
            self.groq_client   = None
            logger.warning(f"LLM not available — using keyword-based fallbacks: {e}")

        # ── Learning path generator ────────────────────────────────────────
        try:
            from modules.learning_path_generator import LearningPathGenerator
            self.learning_path_gen = LearningPathGenerator(config)
            logger.info("✅ Learning path generator loaded")
        except Exception as e:
            self.learning_path_gen = None
            logger.warning(f"Learning path generator not loaded: {e}")

        # ── Ensure output folders ──────────────────────────────────────────
        for folder in [config.UPLOAD_FOLDER, config.RESULTS_FOLDER,
                       config.ANALYTICS_FOLDER, config.LEARNING_PATHS_FOLDER]:
            os.makedirs(folder, exist_ok=True)

        logger.info("✅ ResumeScreeningEngine ready\n")

    # ══════════════════════════════════════════════════════════════════════
    #  MAIN SCREENING METHOD
    # ══════════════════════════════════════════════════════════════════════
    def screen_resumes(self, jd_text, resume_files, job_title="Custom Job", max_resumes=10):
        """Screen multiple resumes against a JD. Returns (screening_id, output_dict)."""
        if not jd_text or not jd_text.strip():
            raise ValueError("Job description text is required")
        if not resume_files:
            raise ValueError("At least one resume file is required")

        resume_files = [f for f in resume_files if f and f.filename][:max_resumes]
        if not resume_files:
            raise ValueError("No valid resume files provided")

        screening_id    = str(uuid.uuid4())
        required_skills = (self._extract_skills_with_llm(jd_text)
                        if self.llm_available
                        else self._extract_skills_basic(jd_text))

        # ── Encode JD once ────────────────────────────────────────────────
        jd_embedding = self.embedder.encode(jd_text, show_progress_bar=False)

        job = {
            'id'               : screening_id,
            'title'            : job_title,
            'description'      : jd_text,
            'embedding'        : jd_embedding.tolist(),
            'semantic_keywords': self._extract_semantic_keywords(jd_text),
            'required_skills'  : required_skills,
            'created_at'       : datetime.now().isoformat()
        }

        # ── Batch save + parse all resumes ────────────────────────────────
        parsed_resumes = []
        for f in resume_files:
            try:
                fname  = secure_filename(f.filename)
                cid    = str(uuid.uuid4())
                fpath  = os.path.join(self.config.UPLOAD_FOLDER, f"{cid}_{fname}")
                f.save(fpath)
                parsed = self.parser.parse_resume_semantic(fpath)
                text   = parsed.get('text', '').strip()
                parsed_resumes.append((cid, fname, fpath, parsed, text))
            except Exception as e:
                logger.error(f"Failed to save/parse {f.filename}: {e}")

        # ── Batch encode all non-empty resume texts ───────────────────────
        texts_to_encode  = [t for _, _, _, _, t in parsed_resumes if len(t) >= 100]
        batch_embeddings = {}
        if texts_to_encode:
            embs = self.embedder.encode(texts_to_encode, batch_size=8,
                                        show_progress_bar=False)
            idx = 0
            for cid, fname, fpath, parsed, text in parsed_resumes:
                if len(text) >= 100:
                    batch_embeddings[cid] = embs[idx]
                    idx += 1

        # ── Process each resume ───────────────────────────────────────────
        results = []
        for cid, fname, fpath, parsed, text in parsed_resumes:
            r = self._process_single_resume(
                cid, fname, fpath, parsed, text,
                job, jd_embedding,
                precomputed_emb=batch_embeddings.get(cid)
            )
            if r:
                results.append(r)

        # ── Rank ──────────────────────────────────────────────────────────
        ranked = self.ranker.rank_candidates_semantic(results, job)

        # ── ML Fit Prediction + SHAP ──────────────────────────────────────
        if self.fit_predictor:
            try:
                ranked = self.fit_predictor.predict_batch(ranked)
                logger.info(f"✅ ML fit + SHAP done for {len(ranked)} candidates")
            except Exception as e:
                logger.error(f"FitPredictor.predict_batch failed: {e}")

        # # 🚨 FORCE FIX FOR DEMO: Override ML prediction based on score
        # for c in ranked:
        #     score = c.get('overall_score', 0)

        #     if score >= 65:
        #         c['ml_fit_label'] = 'Strong Fit'
        #         c['ml_fit_score'] = max(c.get('ml_fit_score', 70), 70)
            
        #     else:
        #         c['ml_fit_label'] = 'Not Fit'

        # ── Apply decisions ───────────────────────────────────────────────
        for c in ranked:
            c['decision']              = self._apply_decision_rules(c)
            c['learning_path_sent']    = False
            c['skill_test_sent']       = False
            c['skill_test_id']         = None
            c['shap_explanation_text'] = self._build_shap_text(c)

        # ── Post-decision actions per candidate ───────────────────────────
        shortlisted_batch = [] 
        for c in ranked:
            action = c['decision'].get('action')

            # ── SHORTLISTED → no email needed ─────────────────────────────
            if action == 'shortlist':
                logger.info(f"✅ Shortlisted → {c.get('filename')}")
                shortlisted_batch.append(c)
                try:
                    email = c.get('email')
                    if email and '@' in str(email):
                        from modules.interview_scheduler import InterviewScheduler
                        scheduler       = InterviewScheduler(self.config)
                        recruiter_email = os.environ.get('SMTP_USERNAME', 'recruiter@company.com')
                        slot_group_id   = scheduler.create_slots(recruiter_email)
                        base_url        = os.environ.get('BASE_URL', 'http://localhost:5002')

                        scheduler.send_interview_invites(
                            candidate_emails = [email],
                            candidate_names  = [c.get('filename', 'Candidate').replace('.pdf','')],
                            job_title        = job.get('title', 'Position'),
                            slot_group_id    = slot_group_id,
                            base_url         = base_url,
                        )
                        c['interview_invite_sent'] = True
                        logger.info(f"📧 Interview invite sent → {email}")
                    else:
                        logger.warning(f"⚠️ Shortlisted but no email found → {c.get('filename')}")
                        c['interview_invite_sent'] = False
                except Exception as e:
                    logger.error(f"Interview invite failed: {e}")
                    c['interview_invite_sent'] = False

            # ── HOLD → generate skill test + email ────────────────────────
            elif action == 'consider':
                try:
                    from modules.skill_test_generator import generate_skill_test
                    from modules.skill_test_emailer   import send_test_email

                    logger.info(f"⏸️ HOLD candidate — generating skill test → {c.get('filename')}")
                    test = generate_skill_test(c, job)

                    if test:
                        # Save test to tests_store folder
                        tests_folder = os.path.join(self.config.BASE_DIR, 'tests_store')
                        os.makedirs(tests_folder, exist_ok=True)
                        test_path = os.path.join(tests_folder, f"{test['test_id']}.json")
                        with open(test_path, 'w', encoding='utf-8') as f:
                            json.dump(test, f, indent=2)

                        # Email test link to candidate
                        sent = send_test_email(c, test, self.config)
                        c['skill_test_sent'] = sent
                        c['skill_test_id']   = test['test_id']

                        if sent:
                            logger.info(f"📝 Skill test emailed → {c.get('email')}")
                        else:
                            logger.warning(f"📝 Skill test generated but email failed → {c.get('email')}")
                    else:
                        logger.warning(f"⚠️ Skill test generation returned None for {c.get('filename')}")

                except Exception as e:
                    logger.error(f"Skill test failed for {c.get('filename')}: {e}")
                    c['skill_test_sent'] = False

            # ── REJECTED → send learning path email ───────────────────────
            elif action == 'reject':
                try:
                    if self.learning_path_gen:
                        lp   = self.learning_path_gen.generate(c, job)
                        sent = send_learning_path_email(c, job, lp, self.config)
                        c['learning_path_sent'] = sent
                        c['learning_path_url']  = f"/learning-path/{c.get('candidate_id')}"
                        if sent:
                            logger.info(f"📚 Learning path emailed → {c.get('email')}")
                except Exception as e:
                    logger.error(f"Learning path failed for {c.get('filename')}: {e}")
                    c['learning_path_sent'] = False
        if shortlisted_batch:
            try:
                from modules.interview_scheduler import InterviewScheduler
                scheduler       = InterviewScheduler(self.config)
                recruiter_email = os.environ.get('SMTP_USERNAME', 'recruiter@company.com')
                base_url        = os.environ.get('BASE_URL', 'http://localhost:5002')
        
                # Find ONE best slot from next week (checks recruiter's Google Calendar)
                slot_group_id = scheduler.create_slots(recruiter_email)
        
                emails       = []
                names        = []
                resume_texts = []
        
                for sc in shortlisted_batch:
                    email = sc.get('email')
                    if email and '@' in str(email):
                        emails.append(email)
                        names.append(sc.get('filename', 'Candidate'))
                        # Pass resume text so scheduler can extract real name
                        resume_texts.append(sc.get('resume_text', ''))
                        sc['interview_invite_sent'] = True
                    else:
                        logger.warning(f"⚠️ Shortlisted but no email → {sc.get('filename')}")
                        sc['interview_invite_sent'] = False
        
                if emails:
                    scheduler.send_interview_invites(
                        candidate_emails = emails,
                        candidate_names  = names,
                        job_title        = job.get('title', 'Position'),
                        slot_group_id    = slot_group_id,
                        base_url         = base_url,
                        resume_texts     = resume_texts,   # NEW — for real name extraction
                    )
                    logger.info(f"📧 Interview invites sent to {len(emails)} shortlisted candidate(s)")
        
            except Exception as e:
                logger.error(f"Interview invite batch failed: {e}")
                for sc in shortlisted_batch:
                    sc['interview_invite_sent'] = False
        # ── Skill gap analysis ────────────────────────────────────────────
        skill_gap = (self._analyze_skill_gaps_advanced(ranked, job)
                    if self.llm_available
                    else self._analyze_skill_gaps_basic(ranked, job))

        output = {
            'screening_id'      : screening_id,
            'job'               : job,
            'candidates'        : ranked,
            'total'             : len(ranked),
            'created_at'        : datetime.now().isoformat(),
            'skill_gap_analysis': skill_gap,
            'recommendations'   : self._generate_recommendations(ranked, job)
        }

        # ── Save results ──────────────────────────────────────────────────
        result_path = os.path.join(self.config.RESULTS_FOLDER, f"{screening_id}.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

        self._save_to_analytics(output)
        logger.info(f"✅ Screening complete: {len(ranked)} candidates → {screening_id}")
        return screening_id, output
    # ══════════════════════════════════════════════════════════════════════
    #  SINGLE RESUME PROCESSING (uses pre-computed batch embeddings)
    # ══════════════════════════════════════════════════════════════════════

    def _process_single_resume(self, candidate_id, filename, file_path,
                                parsed_data, text, job, jd_embedding,
                                precomputed_emb=None):
        """Process one resume. Uses pre-computed embedding if available."""
        try:
            word_count = len(text.split())
            email      = self._extract_email_from_text(text)
            logger.info(f"📄 {filename} ({word_count} words)")

            # ── Empty file guard ──────────────────────────────────────────
            if len(text) < 100:
                logger.warning(f" {filename}: too short ({len(text)} chars)")
                return self._empty_resume_result(candidate_id, filename,
                                                  file_path, email, job)

            github_username = parsed_data.get('github_username')

            # ── FIX: Robust skill extraction ──────────────────────────────
            # Step 1: keyword match against known vocabulary
            skills_from_vocab = self._extract_skills_from_vocabulary(text)
            # Step 2: extractor (spaCy/NLTK based)
            skills_from_extractor = self.skill_extractor.extract_semantic(text)
            extractor_names = [s['skill'].lower() for s in skills_from_extractor]
            # Step 3: Merge — vocab skills + extractor skills, deduplicated
            all_skills = list(set(skills_from_vocab) | set(extractor_names))
            skill_count = len(all_skills)
            skill_list  = all_skills[:12]
            logger.info(f"   Skills: {skill_count} — {skill_list[:5]}")

            # ── Semantic similarity (use pre-computed if available) ────────
            if precomputed_emb is not None:
                resume_emb = precomputed_emb
            else:
                resume_emb = self.embedder.encode(text, show_progress_bar=False)

            semantic    = self.safe_cosine_similarity(resume_emb, jd_embedding)

            # ── Skill relevance ───────────────────────────────────────────
            skill_match = self._calculate_skill_relevance_from_list(
                skill_list, job['semantic_keywords']
            )

            # ── Education ─────────────────────────────────────────────────
            education = self.parser.extract_education_semantic(text)
            edu_score = self._calculate_education_relevance(education, job['description'])

            # ── Base score formula ────────────────────────────────────────
            # Weights: Semantic 40%, Skills 50%, Education 10%
            base = (semantic * 0.40) + (skill_match * 0.50) + (edu_score * 0.10)

            # Quality bonuses
            if word_count > 500: base += 5
            elif word_count > 300: base += 3
            elif word_count > 150: base += 1

            if skill_count >= 8: base += 5
            elif skill_count >= 5: base += 3
            elif skill_count >= 3: base += 1

            # Boost into realistic range 35–95
            boosted = min(35 + (base * 0.6), 95)
            logger.info(f"   Score: {boosted:.1f}% | Semantic: {semantic:.1f}% | Skills: {skill_match:.1f}%")

            # ── Missing skills ─────────────────────────────────────────────
            missing = (
                self._identify_missing_skills_llm(
                    skill_list, job.get('required_skills', []),
                    text[:1000], job['description'][:1000]
                ) if self.llm_available
                else self._identify_missing_skills_basic(
                    skill_list, job.get('required_skills', [])
                )
            )

            # ── GitHub verification ────────────────────────────────────────
            confidence_bonus = 0
            github_score     = None
            github_result    = None

            if github_username and self.github_verifier:
                try:
                    github_result = self.github_verifier.verify_github(
                        github_username, skill_list
                    )
                    if github_result:
                        github_score = github_result.get('github_score')
                        if github_score:
                            confidence_bonus += github_score * 0.15
                            if self.skill_verifier:
                                sv = self.skill_verifier.verify_skills(
                                    skill_list, github_result
                                )
                                confidence_bonus += sv.get('confidence_bonus', 0)
                except Exception as e:
                    logger.error(f"GitHub verification failed for {github_username}: {e}")

            final_score = min(boosted + confidence_bonus, 100)

            return {
                'candidate_id'       : candidate_id,
                'filename'           : filename,
                'email'              : email,
                'resume_text': text[:3000],
                'semantic_score'     : round(semantic, 1),
                'skill_match_score'  : round(skill_match, 1),
                'skill_relevance'    : round(skill_match, 1),
                'education_score'    : round(edu_score, 1),
                'education_relevance': round(edu_score, 1),
                'github_score'       : round(github_score, 1) if github_score else None,
                'confidence_bonus'   : round(confidence_bonus, 1),
                'overall_score'      : round(boosted, 1),
                'final_score'        : round(final_score, 1),
                'extracted_skills'   : skill_list,
                'missing_skills'     : missing,
                'word_count'         : word_count,
                'skill_count'        : skill_count,
                'github_username'    : github_username,
                'github_verification': github_result,
                'has_github'         : github_username is not None,
                'is_empty_resume'    : False,
                'resume_path'        : file_path,
                'shap_explanation'   : [],        # filled by fit_predictor
                'shap_explanation_text': [],      # filled after predict_batch
            }

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}", exc_info=True)
            return None

    def _empty_resume_result(self, cid, filename, fpath, email, job):
        return {
            'candidate_id'     : cid,
            'filename'         : filename,
            'email'            : email,
            'semantic_score'   : 0,
            'skill_match_score': 0,
            'education_score'  : 50,
            'overall_score'    : 10.0,
            'final_score'      : 10.0,
            'extracted_skills' : [],
            'missing_skills'   : [{'skill': s, 'priority': 'high'}
                                   for s in job.get('required_skills', [])[:5]],
            'word_count'       : 0,
            'skill_count'      : 0,
            'is_empty_resume'  : True,
            'rejection_reason' : 'Empty or unreadable file',
            'github_username'  : None,
            'has_github'       : False,
            'resume_path'      : fpath,
            'shap_explanation' : [],
            'shap_explanation_text': ['Resume file was empty or unreadable'],
        }

    # ══════════════════════════════════════════════════════════════════════
    #  FIX: HUMAN-READABLE SHAP TEXT EXPLANATION
    # ══════════════════════════════════════════════════════════════════════

    def _build_shap_text(self, candidate: dict) -> list:
        """
        Converts SHAP values into plain English sentences.

        Output example:
        [
          "Strong skill match boosted the score significantly",
          "Low GitHub activity reduced confidence",
          "Weak semantic similarity with job description",
          "No GitHub profile found"
        ]
        """
        shap_data = candidate.get('shap_explanation', [])
        texts     = []

        if shap_data:
            # Use real SHAP values
            for item in shap_data[:5]:
                label     = item.get('label', item.get('feature', ''))
                direction = item.get('direction', 'neutral')
                magnitude = item.get('magnitude', 'medium')
                value     = item.get('value', 0)
                impact    = item.get('impact', '')

                if direction == 'positive':
                    if magnitude == 'high':
                        texts.append(f"✅ {label} strongly boosted the score ({impact})")
                    else:
                        texts.append(f"✅ {label} positively contributed ({impact})")
                elif direction == 'negative':
                    if magnitude == 'high':
                        texts.append(f"❌ {label} significantly reduced the score ({impact})")
                    else:
                        texts.append(f"⚠️ {label} slightly reduced the score ({impact})")
                else:
                    texts.append(f"➡️ {label} had neutral impact (value: {value})")
        else:
            # FIX: Fallback — generate explanations from raw scores
            s  = candidate.get('semantic_score', 0)
            sk = candidate.get('skill_match_score', 0)
            ed = candidate.get('education_score', 0)
            gh = candidate.get('github_score')
            ms = candidate.get('missing_skills', [])
            wc = candidate.get('word_count', 0)

            if s >= 70:
                texts.append("✅ Strong semantic match with job description")
            elif s >= 50:
                texts.append("⚠️ Moderate semantic match with job description")
            else:
                texts.append("❌ Weak semantic similarity with job description")

            if sk >= 70:
                texts.append("✅ Good skill alignment with job requirements")
            elif sk >= 50:
                texts.append("⚠️ Partial skill match — some gaps identified")
            else:
                texts.append("❌ Low skill match score reduced ranking")

            if ms:
                top_missing = [m['skill'] if isinstance(m, dict) else m for m in ms[:3]]
                texts.append(f"❌ Missing key skills: {', '.join(top_missing)}")

            if gh:
                texts.append(f"✅ GitHub activity verified — confidence boosted (+{gh * 0.15:.1f}%)")
            elif candidate.get('github_username'):
                texts.append("⚠️ GitHub profile found but activity is low")
            else:
                texts.append("❌ No GitHub profile — could not verify coding activity")

            if ed >= 65:
                texts.append("✅ Education background is relevant to this role")
            else:
                texts.append("⚠️ Education relevance to this role is limited")

            if wc > 400:
                texts.append("✅ Detailed resume with sufficient content")
            elif wc < 150:
                texts.append("❌ Resume is very brief — lacks detail")

        return texts[:5]

    # ══════════════════════════════════════════════════════════════════════
    #  FIX: ROBUST SKILL EXTRACTION
    # ══════════════════════════════════════════════════════════════════════

    def _extract_skills_from_vocabulary(self, text: str) -> list:
        """
        Match resume text against known skill vocabulary.
        More robust than pure keyword matching — handles multi-word skills.
        """
        text_lower = text.lower()
        found      = []
        for skill in SKILL_VOCABULARY:
            # Use word boundary for single words, direct search for phrases
            if ' ' in skill:
                if skill in text_lower:
                    found.append(skill)
            else:
                if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                    found.append(skill)
        return found

    # ══════════════════════════════════════════════════════════════════════
    #  DECISION RULES
    # ══════════════════════════════════════════════════════════════════════

    def _apply_decision_rules(self, candidate):
        ml_score      = candidate.get('ml_fit_score')
        overall_score = candidate.get('overall_score', 0)

        # ── Combine ML score + overall score into one final score ─────────
        # If ML available: 60% ML + 40% overall score
        # If ML not available: 100% overall score
        if ml_score is not None:
            final_score = (ml_score * 0.60) + (overall_score * 0.40)
            score_source = 'ML + Semantic (combined)'
        else:
            final_score  = overall_score
            score_source = 'Rule-based (ML unavailable)'

        # Round for display
        final_score = round(final_score, 1)

        # ── Store combined score back on candidate ─────────────────────────
        candidate['combined_score'] = final_score
        candidate['overall_score'] = final_score

        # ── Decision based on combined score ──────────────────────────────
        if final_score >= 65:
            return {
                'action'      : 'shortlist',
                'display'     : '✅ SHORTLISTED',
                'color'       : '#4caf50',
                'message'     : f'Combined score: {final_score}%',
                'next_step'   : 'interview_invite',
                'score_source': score_source,
                'final_score' : final_score
            }
        
        else:
            return {
                'action'      : 'reject',
                'display'     : '❌ REJECTED',
                'color'       : '#f44336',
                'message'     : f'Combined score: {final_score}%',
                'next_step'   : 'learning_path',
                'score_source': score_source,
                'final_score' : final_score,
                'missing_skills': candidate.get('missing_skills', [])
            }

    # ══════════════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def safe_cosine_similarity(self, e1, e2) -> float:
        try:
            import torch
            t1 = (torch.tensor(e1) if not isinstance(e1, torch.Tensor) else e1).float()
            t2 = (torch.tensor(e2) if not isinstance(e2, torch.Tensor) else e2).float()
            if t1.dim() == 1: t1 = t1.unsqueeze(0)
            if t2.dim() == 1: t2 = t2.unsqueeze(0)
            return float(torch.nn.functional.cosine_similarity(t1, t2).item() * 100)
        except Exception as e:
            logger.error(f"Cosine similarity error: {e}")
            return 50.0

    def _calculate_skill_relevance_from_list(self, skill_list: list, jd_keywords: list) -> float:
        """Calculate skill relevance using embedding similarity."""
        if not jd_keywords or not skill_list:
            return 60.0
        st = ' '.join(skill_list)
        kt = ' '.join(jd_keywords)
        if not st.strip() or not kt.strip():
            return 60.0
        try:
            s_emb = self.embedder.encode(st, show_progress_bar=False)
            k_emb = self.embedder.encode(kt, show_progress_bar=False)
            return min(self.safe_cosine_similarity(s_emb, k_emb) + 15, 100.0)
        except Exception as e:
            logger.error(f"Skill relevance calculation error: {e}")
            return 60.0

    def _calculate_education_relevance(self, education: list, jd_text: str) -> float:
        if not education:
            return 50.0
        et = ' '.join([e.get('degree', '') for e in education]).strip()
        if not et:
            return 50.0
        try:
            e_emb = self.embedder.encode(et, show_progress_bar=False)
            j_emb = self.embedder.encode(jd_text[:500], show_progress_bar=False)
            return self.safe_cosine_similarity(e_emb, j_emb)
        except Exception as e:
            logger.error(f"Education relevance error: {e}")
            return 50.0

    def _extract_semantic_keywords(self, text: str) -> list:
        freq = {}
        for w in text.lower().split():
            if len(w) > 3:
                freq[w] = freq.get(w, 0) + 1
        return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:20]]

    def _extract_email_from_text(self, text: str):
        patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
            r'E[\s]*mail[\s]*:[\s]*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})',
        ]
        for p in patterns:
            matches = re.findall(p, text, re.IGNORECASE)
            if matches:
                e = matches[0].strip().lower()
                if '@' in e and '.' in e.split('@')[1]:
                    return e
        return None

    def _extract_skills_with_llm(self, jd_text: str) -> list:
        try:
            prompt = f"List key technical skills from this job description as comma-separated only:\n{jd_text[:800]}"
            response = self.groq_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
            )
            text   = response.choices[0].message.content.strip()
            skills = [s.strip() for s in text.split(',') if s.strip()]
            return skills[:10]
        except Exception as e:
            logger.warning(f"LLM skill extraction failed: {e}")
        return self._extract_skills_basic(jd_text)
    def _extract_skills_basic(self, jd_text: str) -> list:
        text_lower = jd_text.lower()
        return [s for s in SKILL_VOCABULARY if s in text_lower][:10]

    def _identify_missing_skills_llm(self, cs, rs, rt, jt) -> list:
        try:
            prompt = (f"Required: {', '.join(rs[:5])}\n"
                      f"Candidate has: {', '.join(cs[:5])}\n"
                      f"List TOP 3 missing skills as comma-separated only. If none, say None.")
            response = self.groq_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3,
            )
            t = response.choices[0].message.content.strip()
            if t.lower() == 'none':
                return []
            return [
                {'skill': s.strip(), 'priority': 'high' if i < 2 else 'medium', 'context': 'LLM'}
                for i, s in enumerate(t.split(',')) if s.strip()
            ][:3]
        except Exception as e:
            logger.warning(f"LLM missing skill identification failed: {e}")
        return self._identify_missing_skills_basic(cs, rs)

    def _identify_missing_skills_basic(self, cs: list, rs: list) -> list:
        if not rs:
            return []
        cl = [s.lower() for s in cs]
        return [
            {'skill': s, 'priority': 'high' if i < 3 else 'medium', 'context': 'keyword'}
            for i, s in enumerate(rs)
            if not any(s.lower() in c or c in s.lower() for c in cl)
        ][:5]

    def _analyze_skill_gaps_basic(self, candidates: list, job: dict) -> dict:
        from collections import Counter
        all_m = []
        for c in candidates:
            all_m.extend([m['skill'] if isinstance(m, dict) else m
                          for m in c.get('missing_skills', [])])
        avg = {}
        if candidates:
            avg = {
                'semantic': round(sum(c['semantic_score'] for c in candidates) / len(candidates), 1),
                'skill'   : round(sum(c['skill_match_score'] for c in candidates) / len(candidates), 1),
                'overall' : round(sum(c['overall_score'] for c in candidates) / len(candidates), 1),
            }
        return {
            'most_missing_skills': dict(Counter(all_m).most_common(5)),
            'average_scores'     : avg,
            'recommendations'    : []
        }

    def _analyze_skill_gaps_advanced(self, candidates: list, job: dict) -> dict:
        gap = self._analyze_skill_gaps_basic(candidates, job)
        if not self.llm_available or not candidates:
            return gap
        try:
            top_missing = ', '.join(list(gap['most_missing_skills'].keys())[:3])
            prompt      = (f"Job: {job['title']}\nCandidates: {len(candidates)}\n"
                           f"Most missing skills: {top_missing}\n"
                           f"Give 3 concise hiring recommendations as bullet points.")
            response = self.groq_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.5,
            )
            text  = response.choices[0].message.content.strip()
            lines = text.split('\n')
            gap['recommendations']   = [
                l.strip('-• ') for l in lines
                if l.strip() and (l.startswith('-') or l.startswith('•'))
            ][:3]
            gap['detailed_analysis'] = text
        except Exception as e:
            logger.warning(f"LLM skill gap analysis failed: {e}")
        return gap

    def _generate_recommendations(self, candidates: list, job: dict) -> dict:
        top = sorted(candidates, key=lambda x: x['overall_score'], reverse=True)[:3]
        rec = {
            'top_candidates': [
                {'filename': c['filename'], 'score': c['overall_score'],
                 'decision': c['decision']['action']}
                for c in top
            ]
        }
        top_score = candidates[0]['overall_score'] if candidates else 0
        if top_score >= 75:
            rec['hiring_recommendation'] = 'Proceed with top candidate immediately'
        elif top_score >= 60:
            rec['hiring_recommendation'] = 'Schedule interviews with top 2-3 candidates'
        else:
            rec['hiring_recommendation'] = 'No strong candidates found — consider reposting'
        return rec

    def _save_to_analytics(self, data: dict):
        path = os.path.join(self.config.ANALYTICS_FOLDER, 'all_screenings.json')
        analytics = {'screenings': []}
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    analytics = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Could not read analytics file: {e}")

        avg = (round(sum(c['overall_score'] for c in data['candidates'])
                     / len(data['candidates']), 1)
               if data['candidates'] else 0)

        analytics['screenings'].append({
            'screening_id'    : data['screening_id'],
            'job_title'       : data['job']['title'],
            'date'            : data['created_at'],
            'total_candidates': data['total'],
            'average_score'   : avg,
            'top_score'       : data['candidates'][0]['overall_score']
                                if data['candidates'] else 0,
            'skill_gaps'      : data['skill_gap_analysis']['most_missing_skills']
        })
        analytics['screenings'] = analytics['screenings'][-50:]

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(analytics, f, indent=2)
        except IOError as e:
            logger.error(f"Could not save analytics: {e}")

    # ══════════════════════════════════════════════════════════════════════
    #  DATA ACCESS
    # ══════════════════════════════════════════════════════════════════════

    def get_screening_result(self, screening_id: str):
        path = os.path.join(self.config.RESULTS_FOLDER, f"{screening_id}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Could not load screening {screening_id}: {e}")
            return None

    def get_all_screenings(self) -> list:
        if not os.path.exists(self.config.RESULTS_FOLDER):
            return []
        screenings = []
        for fname in os.listdir(self.config.RESULTS_FOLDER):
            if not fname.endswith('.json'):
                continue
            try:
                with open(os.path.join(self.config.RESULTS_FOLDER, fname),
                          'r', encoding='utf-8') as f:
                    data = json.load(f)
                screenings.append({
                    'id'        : data['screening_id'],
                    'job_title' : data['job']['title'],
                    'total'     : data['total'],
                    'created_at': data['created_at'],
                    'top_score' : data['candidates'][0]['overall_score']
                                  if data['candidates'] else 0,
                    'skill_gaps': data.get('skill_gap_analysis', {})
                                      .get('most_missing_skills', {})
                })
            except (json.JSONDecodeError, KeyError, IOError) as e:
                logger.error(f"Could not read screening file {fname}: {e}")
        screenings.sort(key=lambda x: x['created_at'], reverse=True)
        return screenings

    def get_recent_jds(self) -> list:
        jds = []
        if not os.path.exists(JD_STORE):
            return jds
        for fname in os.listdir(JD_STORE):
            if not fname.endswith('.txt'):
                continue
            try:
                with open(os.path.join(JD_STORE, fname), 'r', encoding='utf-8') as f:
                    content = f.read()
                name   = fname.replace('jd_', '').replace('.txt', '')
                parts  = name.split('_')
                display = (' '.join(parts[:-1])
                           if len(parts) > 1 and (parts[-1].isdigit()
                                                   or len(parts[-1]) == 14)
                           else ' '.join(parts))
                jds.append({
                    'filename'    : fname,
                    'display_name': display,
                    'content'     : content,
                    'path'        : os.path.join(JD_STORE, fname)
                })
            except IOError as e:
                logger.error(f"Could not read JD {fname}: {e}")
        jds.sort(key=lambda x: x['filename'], reverse=True)
        return jds

    def get_analytics_summary(self) -> dict:
        path = os.path.join(self.config.ANALYTICS_FOLDER, 'all_screenings.json')
        empty = {'total_screenings': 0, 'total_candidates': 0,
                 'average_score': 0, 'common_skill_gaps': {}}
        if not os.path.exists(path):
            return empty
        try:
            with open(path, 'r', encoding='utf-8') as f:
                a = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Could not load analytics: {e}")
            return empty

        total = sum(s['total_candidates'] for s in a['screenings'])
        avg   = (sum(s['average_score'] for s in a['screenings']) / len(a['screenings'])
                 if a['screenings'] else 0)
        gaps  = {}
        for s in a['screenings']:
            for skill, count in s.get('skill_gaps', {}).items():
                gaps[skill] = gaps.get(skill, 0) + count
        return {
            'total_screenings' : len(a['screenings']),
            'total_candidates' : total,
            'average_score'    : round(avg, 1),
            'common_skill_gaps': dict(sorted(gaps.items(),
                                             key=lambda x: x[1],
                                             reverse=True)[:5])
        }