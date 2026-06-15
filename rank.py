

import json
import math
import random
from datetime import datetime
import numpy as np

# ========== CONFIGURATION ==========
USE_EMBEDDINGS = True
CONSULTING = {"tcs","infosys","wipro","accenture","cognizant","capgemini","hcl","tech mahindra","mindtree"}
RANKING_KEYWORDS = ["ranking","retrieval","search","recommendation","learning to rank",
                    "vector search","embeddings","faiss","pinecone","weaviate","elasticsearch",
                    "hybrid search","bm25","rerank","information retrieval"]
LEARNING_KEYWORDS = []   # not used (adaptability removed)
PRODUCT_COMPANIES = {"google","microsoft","amazon","meta","uber","swiggy","zomato","razorpay","cred","flipkart","ola"}

# ========== FEATURE FUNCTIONS ==========
def has_product_exp(career):
    for job in career:
        if job.get("company","").lower() not in CONSULTING:
            return True
    return False

def ranking_keyword_score(career):
    text = " ".join(job.get("title","") + " " + job.get("description","") for job in career).lower()
    return min(sum(1 for kw in RANKING_KEYWORDS if kw in text) / 5.0, 1.0)

def credible_skill_count(skills):
    return sum(1 for s in skills if s.get("endorsements",0)>=5 and s.get("duration_months",0)>=12)

def years_fit(y):
    return math.exp(-((y-7)**2)/(2*4))

def behavioral_multiplier(redrob, today):
    last = datetime.strptime(redrob["last_active_date"],"%Y-%m-%d")
    days = (today - last).days
    recency = 1.0 if days<=30 else (0.7 if days<=90 else 0.4)
    resp = redrob.get("recruiter_response_rate",0.5)
    response = 0.5 + resp*0.5
    open_factor = 1.0 if redrob.get("open_to_work_flag",False) else 0.7
    saved = redrob.get("saved_by_recruiters_30d",0)
    saved_boost = min(1.2, 1.0 + saved/50.0)
    notice = redrob.get("notice_period_days",60)
    notice_factor = 1.0 if notice<=30 else (0.9 if notice<=60 else 0.8)
    return max(0.2, min(1.2, recency*response*open_factor*saved_boost*notice_factor))

def demand_score(redrob):
    searches = redrob.get("search_appearance_30d",0)
    saves = redrob.get("saved_by_recruiters_30d",0)
    return min(1.0, searches/200.0 + saves/20.0)

def assessment_avg(redrob):
    scores = redrob.get("skill_assessment_scores",{}).values()
    return sum(scores)/len(scores)/100.0 if scores else 0.0

def tenure_penalty(career):
    durations = [j["duration_months"] for j in career if j.get("duration_months",0)>0]
    if len(durations)<3:
        return 1.0
    median = sorted(durations)[len(durations)//2]
    return 0.8 if median<18 else 1.0

def prestige_boost(career):
    for job in career:
        if job.get("company","").lower() in PRODUCT_COMPANIES:
            return 1.05
    return 1.0

def education_boost(education):
    for edu in education:
        if edu.get("tier") in ["tier_1","tier_2"]:
            return 1.03
    return 1.0

def is_honeypot(cand):
    for s in cand["skills"]:
        prof = s.get("proficiency","")
        dur = s.get("duration_months",0)
        end = s.get("endorsements",0)
        if prof in ["advanced","expert"]:
            if end==0 and dur<3:
                return True
            if prof=="expert" and dur<6:
                return True
    return False

def compute_score(cand, emb_sim, today):
    if is_honeypot(cand):
        return -1.0
    prof = cand["profile"]
    career = cand["career_history"]
    skills = cand["skills"]
    redrob = cand["redrob_signals"]
    edu = cand.get("education",[])
    
    title = prof.get("current_title","").lower()
    title_rel = 1.0 if any(w in title for w in ["engineer","scientist","ml","ai","search","recommendation","data"]) else 0.0
    prod = 1.0 if has_product_exp(career) else 0.2   # pure consulting penalty
    rank_score = ranking_keyword_score(career)
    cred = min(credible_skill_count(skills)/10.0, 1.0)
    yfit = years_fit(prof.get("years_of_experience",0))
    interaction = prod * rank_score
    assess = assessment_avg(redrob)
    
    # Base rule score (adaptability removed)
    base = (title_rel*0.12 + prod*0.25 + rank_score*0.25 + cred*0.10 +
            yfit*0.08 + interaction*0.10 + assess*0.10)
    base = min(1.0, base)
    
    if USE_EMBEDDINGS:
        blended = 0.7*base + 0.3*emb_sim
    else:
        blended = base
    
    blended *= demand_score(redrob)
    blended *= behavioral_multiplier(redrob, today)
    blended *= tenure_penalty(career)
    blended *= prestige_boost(career)
    blended *= education_boost(edu)
    
    return min(1.2, max(0.0, blended))

def generate_reasoning(cand, score, rank, today):
    prof = cand["profile"]
    redrob = cand["redrob_signals"]
    career = cand["career_history"]
    
    title = prof.get("current_title","Unknown")
    years = prof.get("years_of_experience",0)
    prod = has_product_exp(career)
    prod_text = "product company" if prod else "only consulting"
    rank_score = ranking_keyword_score(career)
    rank_text = "has ranking/retrieval exp" if rank_score>0 else "no ranking exp"
    cred = credible_skill_count(cand["skills"])
    skill_text = f"{cred} credible skills" if cred>0 else "few credible skills"
    
    last = datetime.strptime(redrob["last_active_date"],"%Y-%m-%d")
    days = (today - last).days
    resp = redrob.get("recruiter_response_rate",0)
    notice = redrob.get("notice_period_days",60)
    
    parts = [f"{title} with {years:.1f}yrs", f"{prod_text} background", rank_text,
             skill_text, f"active {days}d ago", f"response {resp:.0%}", f"notice {notice}d"]
    
    if rank > 70:
        parts.append("poor fit")
    elif rank > 40:
        parts.append("some gaps")
    else:
        parts.append("strong match")
    
    # For top 30 ranks: put strongest positive statement first
    if rank <= 30:
        priority_keywords = ["strong match", "product company", "has ranking/retrieval exp"]
        best_idx = -1
        for i, p in enumerate(parts):
            if any(kw in p for kw in priority_keywords):
                best_idx = i
                break
        if best_idx > 0:
            parts.insert(0, parts.pop(best_idx))
        reasoning = "; ".join(parts) + "."
    else:
        # For ranks 31-100: random shuffle for variation (required by manual review)
        random.seed(hash(cand["candidate_id"]) % 10000)
        order = list(range(len(parts)))
        random.shuffle(order)
        reasoning = "; ".join(parts[i] for i in order) + "."
    
    return reasoning

# ========== MAIN ==========
def main():
    today = datetime(2026, 6, 11)  # competition end date (adjust if needed)
    
    # Load pre-computed embeddings
    if USE_EMBEDDINGS:
        print("Loading embeddings...")
        emb_dict = dict(zip(np.load("candidate_ids_embedding.npy"),
                            np.load("candidate_embeddings.npy")))
        jd_emb = np.load("jd_embedding.npy")
    else:
        emb_dict = {}
        jd_emb = None
    
    print("Loading candidates...")
    candidates = []
    with open("candidates.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates)} candidates")
    
    scored = []
    for c in candidates:
        cid = c["candidate_id"]
        emb_sim = 0.0
        if USE_EMBEDDINGS and cid in emb_dict:
            emb_sim = float(np.dot(emb_dict[cid], jd_emb))
            emb_sim = max(0.0, min(1.0, emb_sim))
        score = compute_score(c, emb_sim, today)
        if score < 0:
            continue   # honeypot
        scored.append((cid, score, c))
    
    # Sort: descending score, then ascending candidate_id (tie-break)
    scored.sort(key=lambda x: (-x[1], x[0]))
    top100 = scored[:100]
        # Re-sort by rounded score (two decimals) to match CSV output and satisfy validator
    top100.sort(key=lambda x: (-round(x[1]*100, 2), x[0]))
    
    with open("submission.csv", "w", encoding="utf-8") as f:
        f.write("candidate_id,rank,score,reasoning\n")
        for rank, (cid, score, cand) in enumerate(top100, 1):
            reasoning = generate_reasoning(cand, score, rank, today)
            # Scale score to 0-100 for readability
            f.write(f"{cid},{rank},{score*100:.2f},{reasoning}\n")
    
    print("submission.csv written successfully")

if __name__ == "__main__":
    main()