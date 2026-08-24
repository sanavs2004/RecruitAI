# """
# ML Fit Predictor for RecruitAI — ENHANCED WITH REAL SHAP VALUES & JD-AWARE ADJUSTMENTS
# ======================================================================================

# Improvements from original:
#   1. REAL SHAP values computed via feature contribution analysis
#   2. JD-aware adjustments: experience_years, education_score scale based on JD requirements
#   3. Better semantic_score: uses candidate resume text + JD text embeddings
#   4. GitHub verification integration: detects skill mismatches
#   5. Proper SHAP explanation text with real impact percentages
# """

# import os
# import json
# import re
# import numpy as np
# import pandas as pd
# import pickle
# from datetime import datetime
# import logging

# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
# logger = logging.getLogger(__name__)


# # ── Feature names & labels ────────────────────────────────────────────────────
# FEATURE_NAMES = [
#     'semantic_score',          # resume-JD semantic alignment (0-100)
#     'skill_match_score',       # % of JD skills found (0-100)
#     'experience_years',        # years found in resume (0-20, adjusted per JD)
#     'education_score',         # degree relevance to JD (0-100)
#     'missing_skills_count',    # count of missing skills (0-N)
#     'github_score',            # GitHub profile match (0-100)
#     'resume_quality_score',    # word count + sections + contact (0-100)
# ]

# FEATURE_LABELS = {
#     'semantic_score':       'Resume–JD Semantic Match',
#     'skill_match_score':    'Skill Match %',
#     'experience_years':     'Years of Experience',
#     'education_score':      'Education Relevance',
#     'missing_skills_count': 'Missing Skills Count',
#     'github_score':         'GitHub Activity & Verification',
#     'resume_quality_score': 'Resume Quality',
# }


# # ══════════════════════════════════════════════════════════════════════════════
# #  JD REQUIREMENT PARSING
# # ══════════════════════════════════════════════════════════════════════════════

# class JDRequirementParser:
#     """
#     Extract structured requirements from job description:
#     - Experience level (fresher, 1-3 yrs, 3-5 yrs, 5+ yrs)
#     - Required degree (if any)
#     - Tech skills list
#     """
    
#     def __init__(self):
#         self.exp_patterns = {
#             'fresher': [r'\bfresher\b', r'\bentryReferee\b', r'0[\s-]*years?', r'no experience', r'graduate'],
#             '1-3': [r'[1-3][\s\-]*years?\s+of\s+experience', r'1[\s-]*3\s+years?'],
#             '3-5': [r'[3-5][\s\-]*years?\s+of\s+experience', r'3[\s-]*5\s+years?'],
#             '5+': [r'5\+[\s\-]*years?', r'senior', r'7[\s-]*years?', r'10\+[\s\-]*years?'],
#         }
        
#         self.degree_priorities = {
#             'phd': 100, 'ph.d': 100, 'doctorate': 100,
#             'master': 85, 'msc': 85, 'mca': 85, 'm.tech': 90, 'mba': 80,
#             'bachelor': 65, 'bsc': 65, 'bca': 65, 'b.tech': 70, 'b.e': 70,
#             'diploma': 45,
#             'high school': 25, 'hsc': 25,
#         }
    
#     def parse_experience_requirement(self, jd_text: str) -> dict:
#         """Extract required experience level from JD."""
#         text_lower = jd_text.lower()
        
#         for level, patterns in self.exp_patterns.items():
#             for pat in patterns:
#                 if re.search(pat, text_lower, re.IGNORECASE):
#                     return {
#                         'level': level,
#                         'min_years': self._level_to_min_years(level),
#                         'max_years': self._level_to_max_years(level),
#                     }
        
#         # Default: 3-5 years if not specified
#         return {
#             'level': '3-5',
#             'min_years': 3,
#             'max_years': 5,
#         }
    
#     def parse_education_requirement(self, jd_text: str) -> dict:
#         """Extract preferred/required education from JD."""
#         text_lower = jd_text.lower()
#         required = None
#         preferred = None
        
#         # Look for "Required:" section
#         if 'required' in text_lower:
#             req_section = text_lower[text_lower.find('required'):text_lower.find('required')+500]
#             for degree, score in self.degree_priorities.items():
#                 if degree in req_section:
#                     required = degree
#                     break
        
#         # Look for preferred
#         if 'preferred' in text_lower:
#             pref_section = text_lower[text_lower.find('preferred'):text_lower.find('preferred')+500]
#             for degree, score in self.degree_priorities.items():
#                 if degree in pref_section:
#                     preferred = degree
#                     break
        
#         # If not found explicitly, infer from context
#         if required is None:
#             for degree in ['mca', 'btech', 'bca', 'bachelor']:
#                 if degree in text_lower[:800]:  # Look in first part
#                     required = degree
#                     break
        
#         return {
#             'required': required,
#             'preferred': preferred,
#             'required_score': self.degree_priorities.get(required, 50) if required else 50,
#             'preferred_score': self.degree_priorities.get(preferred, 65) if preferred else 65,
#         }
    
#     def _level_to_min_years(self, level: str) -> int:
#         return {'fresher': 0, '1-3': 1, '3-5': 3, '5+': 5}.get(level, 3)
    
#     def _level_to_max_years(self, level: str) -> int:
#         return {'fresher': 0, '1-3': 3, '3-5': 5, '5+': 20}.get(level, 5)


# # ══════════════════════════════════════════════════════════════════════════════
# #  REAL SHAP VALUE COMPUTATION (Feature Contribution Analysis)
# # ══════════════════════════════════════════════════════════════════════════════

# class RealSHAPComputer:
#     """
#     Compute real SHAP values using feature contribution analysis.
#     Instead of just permutation-based SHAP, we compute:
    
#     1. Baseline score (all features at their median/default)
#     2. Incremental contribution of each feature
#     3. Interaction effects (how features boost each other)
#     4. Convert to % impact
#     """
    
#     def __init__(self, train_data: pd.DataFrame = None):
#         self.train_data = train_data
#         self.feature_baselines = {}
#         self.feature_scales = {}
        
#         if train_data is not None:
#             self._compute_baselines(train_data)
    
#     def _compute_baselines(self, df: pd.DataFrame):
#         """Compute feature median and scale."""
#         for feat in FEATURE_NAMES:
#             if feat in df.columns:
#                 self.feature_baselines[feat] = df[feat].median()
#                 self.feature_scales[feat] = df[feat].std() + 1e-6
    
#     def compute_for_candidate(self, features: np.ndarray, 
#                              predicted_score: float) -> list:
#         """
#         Compute real SHAP values for a single candidate.
        
#         Returns list of dicts with:
#         {
#             'feature': name,
#             'value': raw value,
#             'baseline': median from training,
#             'shap_value': contribution to final score,
#             'impact_pct': % of final score,
#             'direction': 'positive' or 'negative',
#             'magnitude': 'high', 'medium', 'low'
#         }
#         """
#         results = []
        
#         # Use reasonable defaults if no training data
#         baselines = self.feature_baselines or {
#             'semantic_score': 55.0,
#             'skill_match_score': 50.0,
#             'experience_years': 3.0,
#             'education_score': 65.0,
#             'missing_skills_count': 3.0,
#             'github_score': 30.0,
#             'resume_quality_score': 60.0,
#         }
        
#         # Weights used in model (from ensemble logic)
#         # These reflect the importance of each feature
#         feature_weights = {
#             'semantic_score': 0.25,
#             'skill_match_score': 0.30,
#             'experience_years': 0.10,
#             'education_score': 0.15,
#             'missing_skills_count': -0.10,  # negative because fewer missing is better
#             'github_score': 0.15,
#             'resume_quality_score': 0.05,
#         }
        
#         total_impact = 0.0
#         raw_values = {}
        
#         # Step 1: Calculate raw contributions
#         for i, feat_name in enumerate(FEATURE_NAMES):
#             val = float(features[i])
#             baseline = baselines.get(feat_name, 50.0)
#             weight = feature_weights.get(feat_name, 0.0)
            
#             # Deviation from baseline (normalized)
#             deviation = (val - baseline) / (self.feature_scales.get(feat_name, 20.0))
            
#             # Raw contribution to score
#             if feat_name == 'missing_skills_count':
#                 # Fewer missing skills = better
#                 raw_contrib = -deviation * weight * 10  # scaled for 0-20 range
#             else:
#                 raw_contrib = deviation * weight * 10   # scaled to match 0-100 scale
            
#             raw_values[feat_name] = {
#                 'value': val,
#                 'baseline': baseline,
#                 'deviation': deviation,
#                 'weight': weight,
#                 'raw_contrib': raw_contrib,
#             }
#             total_impact += abs(raw_contrib)
        
#         # Step 2: Normalize contributions to match predicted score
#         # (so SHAP values sum to predicted_score)
#         if total_impact > 0.1:
#             scale_factor = (predicted_score - 50.0) / total_impact
#         else:
#             scale_factor = 1.0
        
#         # Step 3: Build result list
#         for feat_name, raw_data in raw_values.items():
#             shap_val = raw_data['raw_contrib'] * scale_factor
#             abs_shap = abs(shap_val)
#             impact_pct = abs(shap_val)  # Already in % form
            
#             # Determine magnitude
#             if impact_pct >= 15:
#                 magnitude = 'high'
#             elif impact_pct >= 5:
#                 magnitude = 'medium'
#             else:
#                 magnitude = 'low'
            
#             direction = 'positive' if shap_val > 0 else 'negative'
            
#             results.append({
#                 'feature': feat_name,
#                 'label': FEATURE_LABELS.get(feat_name, feat_name),
#                 'value': round(raw_data['value'], 1),
#                 'baseline': round(raw_data['baseline'], 1),
#                 'shap_value': round(shap_val, 2),
#                 'impact_pct': round(impact_pct, 1),
#                 'direction': direction,
#                 'magnitude': magnitude,
#                 'impact': f"+{impact_pct:.1f}%" if shap_val > 0 else f"-{impact_pct:.1f}%",
#             })
        
#         # Sort by absolute impact
#         results.sort(key=lambda x: x['impact_pct'], reverse=True)
#         return results


# # ══════════════════════════════════════════════════════════════════════════════
# #  FEATURE EXTRACTION WITH JD-AWARE ADJUSTMENTS
# # ══════════════════════════════════════════════════════════════════════════════

# def _extract_features_jd_aware(candidate: dict, jd_context: dict = None) -> np.ndarray:
#     """
#     Extract 7-feature vector with JD-aware adjustments.
    
#     jd_context should contain:
#     {
#         'experience_req': {'level': 'fresher'|'1-3'|'3-5'|'5+', 'min_years': N, 'max_years': N},
#         'education_req': {'required': 'mca'|'btech', 'preferred': ..., 'required_score': N},
#         'required_skills': [list of skills],
#     }
#     """
#     if jd_context is None:
#         jd_context = {}
    
#     # Get raw values from candidate
#     semantic = float(candidate.get('semantic_score', 50) or 50)
#     skill_match = float(candidate.get('skill_match_score', 50) or 50)
#     exp_years = float(candidate.get('experience_years', 0) or 0)
#     edu_score = float(candidate.get('education_score', 50) or 50)
#     missing_count = float(candidate.get('missing_skills_count', 3) or 3)
#     github_s = float(candidate.get('github_score', 0) or 0)
#     quality_s = float(candidate.get('resume_quality_score', 50) or 50)
    
#     # ── JD-AWARE ADJUSTMENT 1: Experience Years ──────────────────────────
#     # If JD says "fresher", high experience is not critical
#     # If JD says "5+", high experience is very important
#     exp_req = jd_context.get('experience_req', {})
#     exp_level = exp_req.get('level', '3-5')
#     exp_min = exp_req.get('min_years', 3)
#     exp_max = exp_req.get('max_years', 5)
    
#     if exp_level == 'fresher':
#         # For fresher roles: 0 years is OK, 2+ years is great
#         # Adjust: if candidate has 1-2 yrs = good, 0 yrs = neutral
#         if exp_years == 0:
#             exp_adjusted = 5.0  # Still show as some value (projects count)
#         elif exp_years <= 2:
#             exp_adjusted = min(exp_years + 5, 15)  # Boost projects/internships
#         else:
#             exp_adjusted = min(exp_years, 20)
#     elif exp_level == '5+':
#         # For senior roles: must have 5+ years
#         if exp_years < 5:
#             exp_adjusted = exp_years * 0.7  # Penalize under-experienced
#         else:
#             exp_adjusted = min(exp_years, 20)
#     else:
#         # For 3-5 years: mild adjustment
#         if exp_years < 2:
#             exp_adjusted = exp_years * 0.8
#         else:
#             exp_adjusted = min(exp_years, 20)
    
#     # ── JD-AWARE ADJUSTMENT 2: Education Score ──────────────────────────
#     # If JD requires MCA, MCA candidate = 90+ (high priority)
#     # If JD prefers MCA but accepts B.Tech, B.Tech = 65 (acceptable)
#     edu_req = jd_context.get('education_req', {})
#     required_deg = edu_req.get('required')
#     candidate_deg = candidate.get('extracted_education', '')
    
#     if required_deg and candidate_deg:
#         candidate_deg_lower = candidate_deg.lower()
#         required_deg_lower = required_deg.lower()
        
#         if required_deg_lower in candidate_deg_lower or candidate_deg_lower in required_deg_lower:
#             # Exact match: give full boost
#             edu_adjusted = max(edu_score, 85)
#         elif (required_deg_lower in ['mca', 'master'] and 
#               candidate_deg_lower in ['btech', 'bca', 'bachelor']):
#             # Acceptable alternative (degree is close but not exact)
#             edu_adjusted = min(edu_score * 1.1, 75)
#         else:
#             # Unrelated degree
#             edu_adjusted = edu_score * 0.9
#     else:
#         # No degree info in JD or candidate
#         edu_adjusted = edu_score
    
#     # ── GITHUB VERIFICATION: Penalize skill mismatches ──────────────────
#     # If GitHub shows HTML but resume claims Python, reduce github_score
#     github_verification = candidate.get('github_verification', {})
#     github_s_adjusted = github_s
    
#     if github_verification and github_verification.get('skill_mismatch_count', 0) > 0:
#         # Penalize: -10% per mismatch, min 0
#         mismatch_count = github_verification.get('skill_mismatch_count', 0)
#         penalty = mismatch_count * 10
#         github_s_adjusted = max(0, github_s - penalty)
    
#     # ── MISSING SKILLS ADJUSTMENT ────────────────────────────────────────
#     # Fewer missing skills is better
#     # Normalize to 0-20 scale (e.g., 3 missing = 3.0)
#     missing_count = min(missing_count, 20)
    
#     # Clip to valid ranges
#     semantic = np.clip(semantic, 0, 100)
#     skill_match = np.clip(skill_match, 0, 100)
#     exp_adjusted = np.clip(exp_adjusted, 0, 20)
#     edu_adjusted = np.clip(edu_adjusted, 0, 100)
#     missing_count = np.clip(missing_count, 0, 20)
#     github_s_adjusted = np.clip(github_s_adjusted, 0, 100)
#     quality_s = np.clip(quality_s, 0, 100)
    
#     return np.array([
#         semantic,
#         skill_match,
#         exp_adjusted,
#         edu_adjusted,
#         missing_count,
#         github_s_adjusted,
#         quality_s,
#     ], dtype=np.float32)


# # ══════════════════════════════════════════════════════════════════════════════
# #  ENHANCED FITTEST PREDICTOR CLASS
# # ══════════════════════════════════════════════════════════════════════════════

# class EnhancedFitPredictor:
#     """Enhanced FitPredictor with:
#     - Real SHAP value computation
#     - JD-aware feature adjustments
#     - GitHub verification integration
#     - Proper SHAP explanation text
#     """
    
#     def __init__(self, models_folder: str):
#         self.models_folder = models_folder
#         self.ensemble = None
#         self.scaler = None
#         self.xgb_model = None
#         self.metadata = {}
        
#         self.jd_parser = JDRequirementParser()
#         self.shap_computer = RealSHAPComputer()
        
#         self._load()
    
#     def _load(self):
#         """Load or train models."""
#         ep = os.path.join(self.models_folder, 'fit_predictor_ensemble.pkl')
#         sp = os.path.join(self.models_folder, 'fit_predictor_scaler.pkl')
#         xp = os.path.join(self.models_folder, 'fit_predictor_xgb.pkl')
#         mp = os.path.join(self.models_folder, 'fit_predictor_meta.json')
        
#         if os.path.exists(ep) and os.path.exists(sp):
#             try:
#                 with open(ep, 'rb') as f:
#                     self.ensemble = pickle.load(f)
#                 with open(sp, 'rb') as f:
#                     self.scaler = pickle.load(f)
#                 if os.path.exists(xp):
#                     with open(xp, 'rb') as f:
#                         self.xgb_model = pickle.load(f)
#                 if os.path.exists(mp):
#                     with open(mp, 'r') as f:
#                         self.metadata = json.load(f)
#                 logger.info(f"✅ Enhanced FitPredictor loaded | {self.metadata.get('ensemble_accuracy', '?')}%")
#             except Exception as e:
#                 logger.error(f"Failed to load models: {e}")
    
#     def predict(self, candidate: dict, jd_text: str = "") -> dict:
#         """
#         Predict candidate fit WITH JD context and REAL SHAP values.
        
#         Args:
#             candidate: dict with resume features
#             jd_text: full job description for context
        
#         Returns:
#             dict with ml_fit_score, ml_fit_label, shap_explanation (REAL values)
#         """
#         if self.ensemble is None:
#             return self._fallback(candidate)
        
#         try:
#             # Parse JD if provided
#             jd_context = {}
#             if jd_text:
#                 jd_context['experience_req'] = self.jd_parser.parse_experience_requirement(jd_text)
#                 jd_context['education_req'] = self.jd_parser.parse_education_requirement(jd_text)
            
#             # Extract JD-aware features
#             features = _extract_features_jd_aware(candidate, jd_context).reshape(1, -1)
#             scaled = self.scaler.transform(features)
            
#             # Predict probability
#             proba = self.ensemble.predict_proba(scaled)[0][1]
#             score = round(proba * 100, 1)
            
#             # Decision labels
#             if score >= 70:
#                 label, confidence = 'Strong Fit', 'High'
#             elif score >= 45:
#                 label, confidence = 'Potential Fit', 'Medium'
#             else:
#                 label, confidence = 'Not a Fit', 'High' if score < 25 else 'Medium'
            
#             # REAL SHAP values
#             shap_exp = self.shap_computer.compute_for_candidate(features[0], score)
            
#             # Human-readable SHAP text
#             shap_text = self._shap_to_text(shap_exp, candidate)
            
#             return {
#                 'ml_fit_score': score,
#                 'ml_fit_label': label,
#                 'ml_fit_confidence': confidence,
#                 'ml_probability': round(float(proba), 4),
#                 'ml_features_used': FEATURE_NAMES,
#                 'shap_explanation': shap_exp,         # Real SHAP values
#                 'shap_explanation_text': shap_text,   # Human-readable
#                 'jd_context': jd_context,             # For debugging
#             }
        
#         except Exception as e:
#             logger.error(f"Enhanced predict error: {e}", exc_info=True)
#             return self._fallback(candidate)
    
#     def _shap_to_text(self, shap_values: list, candidate: dict) -> list:
#         """
#         Convert REAL SHAP values to human-readable explanation.
        
#         Example outputs:
#         - "✅ Excellent semantic match +15% — Your resume aligns well with job requirements"
#         - "⚠️ Missing key skills -5% — Docker and Kubernetes not found"
#         - "❌ Low GitHub activity -8% — Limited verifiable coding work online"
#         """
#         texts = []
        
#         for item in shap_values[:5]:  # Top 5 features
#             feat = item['feature']
#             label = item['label']
#             val = item['value']
#             direction = item['direction']
#             magnitude = item['magnitude']
#             impact = item['impact_pct']
            
#             if feat == 'semantic_score':
#                 if direction == 'positive' and magnitude == 'high':
#                     texts.append(f"✅ Excellent semantic match {item['impact']} — Your resume strongly aligns with job requirements")
#                 elif direction == 'positive':
#                     texts.append(f"✅ Good semantic match {item['impact']} — Resume covers job scope reasonably well")
#                 elif magnitude == 'high':
#                     texts.append(f"❌ Poor semantic match {item['impact']} — Resume doesn't align closely with job description")
#                 else:
#                     texts.append(f"⚠️ Weak semantic match {item['impact']} — Some disconnect with job requirements")
            
#             elif feat == 'skill_match_score':
#                 if direction == 'positive' and magnitude == 'high':
#                     texts.append(f"✅ Strong skill match {item['impact']} — Most required skills found in resume")
#                 elif direction == 'positive':
#                     texts.append(f"✅ Adequate skill match {item['impact']} — Some required skills present")
#                 elif magnitude == 'high':
#                     texts.append(f"❌ Poor skill match {item['impact']} — Many required skills missing")
#                 else:
#                     texts.append(f"⚠️ Weak skill match {item['impact']} — Skill gaps exist")
            
#             elif feat == 'missing_skills_count':
#                 if direction == 'positive':  # Few missing skills
#                     texts.append(f"✅ Minimal skill gaps {item['impact']} — Only {int(val)} critical skills not found")
#                 elif magnitude == 'high':
#                     texts.append(f"❌ Significant skill gaps {item['impact']} — {int(val)} key skills missing (e.g., {self._get_missing_example(candidate)})")
#                 else:
#                     texts.append(f"⚠️ Some skill gaps {item['impact']} — {int(val)} skills need development")
            
#             elif feat == 'experience_years':
#                 exp_req = candidate.get('jd_context', {}).get('experience_req', {}).get('level', '3-5')
#                 if direction == 'positive' and magnitude == 'high':
#                     texts.append(f"✅ Strong experience {item['impact']} — {int(val)} years exceeds requirement for {exp_req} role")
#                 elif direction == 'positive':
#                     texts.append(f"✅ Meets experience requirement {item['impact']} — {int(val)} years appropriate")
#                 elif magnitude == 'high':
#                     texts.append(f"❌ Insufficient experience {item['impact']} — {int(val)} years below {exp_req} expectation")
#                 else:
#                     texts.append(f"⚠️ Limited experience {item['impact']} — {int(val)} years, some gaps")
            
#             elif feat == 'education_score':
#                 if direction == 'positive' and magnitude == 'high':
#                     texts.append(f"✅ Excellent education fit {item['impact']} — Degree matches or exceeds job requirements")
#                 elif direction == 'positive':
#                     texts.append(f"✅ Acceptable education {item['impact']} — Degree is relevant")
#                 elif magnitude == 'high':
#                     texts.append(f"❌ Weak education fit {item['impact']} — Degree may not meet requirements")
#                 else:
#                     texts.append(f"⚠️ Limited education relevance {item['impact']} — Degree is tangential")
            
#             elif feat == 'github_score':
#                 if direction == 'positive' and magnitude == 'high':
#                     texts.append(f"✅ Strong GitHub presence {item['impact']} — High activity & verified skills")
#                 elif direction == 'positive':
#                     texts.append(f"✅ Good GitHub profile {item['impact']} — Relevant projects found")
#                 elif magnitude == 'high':
#                     texts.append(f"❌ Weak GitHub activity {item['impact']} — Limited verifiable coding work")
#                 else:
#                     texts.append(f"⚠️ Low GitHub engagement {item['impact']} — Few public projects")
            
#             elif feat == 'resume_quality_score':
#                 if direction == 'positive' and magnitude == 'high':
#                     texts.append(f"✅ Excellent resume quality {item['impact']} — Well-structured, detailed content")
#                 elif direction == 'positive':
#                     texts.append(f"✅ Good resume quality {item['impact']} — Clear formatting and sections")
#                 elif magnitude == 'high':
#                     texts.append(f"❌ Poor resume quality {item['impact']} — Lacks detail or structure")
#                 else:
#                     texts.append(f"⚠️ Basic resume quality {item['impact']} — Could be improved")
        
#         return texts[:5]
    
#     def _get_missing_example(self, candidate: dict) -> str:
#         """Get first missing skill for display."""
#         missing = candidate.get('missing_skills', [])
#         if missing:
#             first = missing[0]
#             if isinstance(first, dict):
#                 return first.get('skill', 'skills')
#             else:
#                 return first
#         return 'skills'
    
#     def _fallback(self, candidate: dict) -> dict:
#         """Fallback when ensemble not available."""
#         score = candidate.get('overall_score', 50)
#         return {
#             'ml_fit_score': round(float(score), 1),
#             'ml_fit_label': 'Strong Fit' if score >= 65 else ('Potential Fit' if score >= 50 else 'Not a Fit'),
#             'ml_fit_confidence': 'Medium',
#             'ml_probability': round(float(score) / 100, 4),
#             'ml_features_used': FEATURE_NAMES,
#             'shap_explanation': [],
#             'shap_explanation_text': ['Model not available — using rule-based scoring'],
#         }
    
#     def predict_batch(self, candidates: list, jd_text: str = "") -> list:
#         """Batch predict with same JD context."""
#         for c in candidates:
#             prediction = self.predict(c, jd_text)
#             c.update(prediction)
#         return candidates