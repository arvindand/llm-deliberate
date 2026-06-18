# Quick Start Guide

Get up and running with LLM Deliberate in minutes. This guide walks through all major features—use it as a tutorial or a feature checklist.

---

## Getting Started

1. Start the app: `./start.sh`
2. Open <http://localhost:5173>
3. Have an OpenRouter API key configured (for automated collection)

---

## Your First Experiment

**Create a new experiment:**

- Name: `My First Experiment`
- Description: `Testing multi-model deliberation`

**Add a question** (this logic puzzle works well for multi-model reasoning):

> Three logicians walk into a bar. The bartender asks "Does everyone want a beer?" The first logician says "I don't know". The second says "I don't know". The third says "Yes". Explain the logic behind their answers. What does each logician want?

- Type: **Reasoning**
- Ground Truth: `All three want a beer`

> **Note:** Auto collection and Deliberation are mutually exclusive per question. Once you collect one-shot responses, you cannot run deliberation on that question (and vice versa). Add multiple questions to try both modes.

---

## Feature Walkthrough

### 1. Response Collection (Auto)

- [ ] Click **Auto** in Responses section
- [ ] Select 3+ models from different providers (e.g., Claude Sonnet, GPT-4o, Gemini)
- [ ] Review estimated cost before starting
- [ ] Click **Collect** → responses stream in with metadata (tokens, latency, cost)

### 2. Multi-Round Deliberation

*(Use a different question - Auto and Deliberate are mutually exclusive)*

- [ ] Add another question, e.g.:
  > "Give two strong arguments for and two against deploying fully-autonomous delivery drones in dense cities."
- [ ] Click **Deliberate** button (not Auto)
- [ ] Select 3 models
- [ ] Set **Maximum Rounds** to 2-3
- [ ] Start → watch real-time progress
- [ ] After completion:
  - [ ] See round tabs (color-coded: blue/purple/amber)
  - [ ] See **Convergent Answer** at top
  - [ ] Toggle between rounds to see how responses evolved

### 3. Evolution Visualization

- [ ] With the deliberated question's multi-round data, toggle **"Evolution"** view
- [ ] See convergence chart showing similarity over rounds
- [ ] See per-model timeline
- [ ] Expand a round to see word-level diffs (if available)

### 4. Rankings Collection

- [ ] Click **Auto** in Rankings section
- [ ] Use **"Use Response Models as Judges"** button
- [ ] Collect rankings → each shows confidence score (0-100%)
- [ ] Verify reasoning is included in ranking metadata

### 5. Agreement Heatmap

- [ ] With 3+ rankings, click **"View Agreement"**
- [ ] See red-yellow-green heatmap matrix
- [ ] Check diversity score (0-1 scale)
- [ ] Hover cells to see pairwise agreement percentages

### 6. Aggregation Methods

- [ ] Click **"Compare Aggregation Methods"**
- [ ] Verify all 8 methods appear:
  1. Plurality
  2. Borda Count
  3. Weighted Borda
  4. Copeland
  5. Ranked Pairs
  6. Schulze
  7. STV/Instant Runoff
  8. Approval Voting
- [ ] Check if any show **"Unanimous"** (all agree)
- [ ] Note any method that produces different winner

### 7. Debate Format (API Only)

- [ ] Open <http://localhost:8000/docs>
- [ ] Use `POST /experiments/{id}/automate/debate`
- [ ] Supply 2 debater models and a different judge model
- [ ] Poll the returned job through the automation status endpoint
- [ ] Verify the result contains arguments and a judge verdict with reasoning

### 8. Chairman Synthesis

- [ ] With rankings collected, click **"Synthesize Final Answer"**
- [ ] Select a chairman model
- [ ] Wait for synthesis → markdown-rendered result
- [ ] Verify it integrates top-ranked responses
- [ ] Reload the page and verify the synthesis is still displayed

### 9. Cost Dashboard

- [ ] Click **"View Costs"** in experiment header
- [ ] Check all 5 tabs:
  - [ ] Overview (total spend)
  - [ ] By Question
  - [ ] By Model
  - [ ] By Round
  - [ ] By Provider

### 10. Export Features

- [ ] Click **Export JSON** → download full experiment
- [ ] Click **Export CSV** → download flattened data
- [ ] On question card, click export icon (JSON)
- [ ] Shift+click export icon (CSV)

---

## Quick Tour

Short on time? Try these core features:

1. **Auto Collect** - 2 models, 1 question
2. **Compare Methods** - see 8 aggregation algorithms
3. **Chairman Synthesis** - turn ranked responses into a persisted final answer

---
