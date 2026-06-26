# Redrob Hackathon – Intelligent Candidate Ranking

## Overview

This ranker scores 100,000 candidates against a job description and outputs the top 100. It uses rule‑based features, behavioral signals, and pre‑computed embeddings (`all-MiniLM-L6-v2`).

## Setup

```bash
pip install -r requirements.txt

## Pre‑compute embeddings (first time only)
bash
python precompute.py
This creates:

jd_embedding.npy

candidate_embeddings.npy

candidate_ids_embedding.npy

Run ranking
bash
python rank.py
This produces submission.csv with the top 100 candidates.

Compute constraints respected:

CPU only (no GPU)

Runs in under 5 minutes on 16 GB RAM

No network calls during ranking

No external API calls (OpenAI, etc.)

##Methodology
The ranker combines 6 key components:

Product experience – penalises candidates with only consulting backgrounds (TCS, Infosys, Wipro, etc.) because the JD explicitly says they are a bad fit.

Career history analysis – reads actual job titles and descriptions to identify candidates who have built ranking, search, or recommendation systems in production. This captures real experience, not just keywords.

Credible skills – only counts skills with endorsements ≥5 and duration ≥12 months. This avoids keyword stuffers who list AI skills without real experience.

Behavioral signals – checks recency (last active date), recruiter response rate, open‑to‑work flag, saved by recruiters, and notice period to confirm the candidate is actually available.

Embedding similarity – uses all-MiniLM-L6-v2 to measure semantic similarity between the candidate's profile and the job description. This captures meaning beyond keywords.

Honeypot detection – kills candidates with impossible skill combinations (e.g., expert proficiency with 0 endorsements or <3 months duration).

##Scoring Formula
text
Base Score = 
  Title Relevance × 0.12
  + Product Experience × 0.25
  + Ranking Experience × 0.25
  + Credible Skills × 0.10
  + Years Fit × 0.08
  + Interaction (Product × Ranking) × 0.10
  + Assessment Average × 0.10

Blended = 0.7 × Base + 0.3 × Embedding Similarity
Final Score = Blended × Behavioral Multiplier
Reasoning Generation
Each candidate in the top 100 gets a 1‑2 sentence reasoning with:

Specific facts from the profile (title, years, skills, activity)

JD connection (product experience, ranking experience)

Honest concerns (long notice period, low response rate, poor fit)

For top 30 ranks: strengths first (strong match, product company, has ranking exp)
For ranks 31‑100: varied order to satisfy manual review requirements

No hallucination – every claim is verified from the candidate's profile.

##Files
File	Purpose
rank.py	Main ranking script
precompute.py	Generates embeddings (run once)
requirements.txt	Python dependencies
submission.csv	Output (top 100 candidates)
validate_submission.py	Format validator
Validation
bash
python validate_submission.py submission.csv


## Sandbox

Live Colab sandbox (runs on a 1000‑candidate sample):
https://colab.research.google.com/drive/1AcAerobUnAc4ZOf2tSHUrSkj_GJh3gtr?usp=sharing

