# LLM Deliberate

**An experimentation tool for exploring multi-model LLM deliberation and aggregation methods.**

> Inspired by [Andrej Karpathy's llm-council](https://github.com/karpathy/llm-council)

![Deliberate UI](docs/screenshot-deliberation.png)

## ⚠️ Maintenance Disclaimer

**This is an experimentation tool and experimental project.** Like Karpathy's original llm-council, this is primarily a "vibe coded" exploration of LLM deliberation methods. While functional and documented, **this project will not be actively maintained**. The code is provided as-is for inspiration, learning, and further experimentation.

Feel free to fork and modify it for your own research needs!

## How Deliberate Differs from llm-council

While inspired by Karpathy's llm-council, Deliberate takes a different approach:

| Feature | llm-council | Deliberate |
| ------- | ----------- | ---------- |
| **Purpose** | Chat interface with synthesized final answer | Experimentation tool for studying aggregation methods |
| **Output** | Single "Chairman" synthesized response | Side-by-side comparison of 5 voting algorithms |
| **Deliberation** | Single round + review | Multi-round iterative refinement |
| **Analysis** | Rankings displayed | Agreement matrices, cost dashboards, export to CSV/JSON |
| **Data Model** | Conversation threads | Structured experiments with questions |
| **Manual Entry** | Not supported | Full support for manual data collection |

Deliberate focuses on **researching the deliberation process itself** rather than producing final answers. It's designed to help answer questions like:

- When do different aggregation methods agree or disagree?
- Does model diversity improve consensus quality?
- What types of questions lead to disagreement among judges?

## Overview

Deliberate is an experimental tool for studying multi-model responses, ranking/judging behavior, and (optionally) **multi-round deliberation with convergence detection**.

You can use it in a few common modes:

- **Responses-only:** collect and compare raw model answers side-by-side
- **Rankings + aggregation:** have models rank each other, then compare formal voting methods on the same set of judgments
- **Multi-round deliberation:** run iterative rounds where models see peer responses and revise their own; the job can stop early if responses converge

For the aggregation layer, Deliberate implements several algorithms from social choice theory:

- **Plurality** — Simple first-place vote counting
- **Borda Count** — Positional voting with points for each rank
- **Weighted Borda** — Borda weighted by judge confidence scores
- **Copeland (Condorcet)** — Pairwise comparison winner
- **Ranked Pairs (Tideman)** — Handles voting cycles gracefully

## Research Questions

This tool helps explore:

1. **Do aggregation methods agree?** When do Borda and Condorcet methods produce different winners?
2. **Does diversity matter?** Is a council of diverse models better than self-consistency from one strong model?
3. **When does consensus fail?** What types of questions lead to disagreement?
4. **Do models show bias?** Do certain models consistently rank others higher or lower? The agreement matrix visualization helps identify clustering patterns among judges.

### What I've Observed

In our experiments, a few patterns emerged:

- **Convergence on logic**: When faced with objective reasoning problems (logic puzzles, math), diverse models tend to converge on the same correct answer and reasoning chain.
- **Herding on simple facts**: Paradoxically, models can become *less* reliable through deliberation on trivial questions—they sometimes over-adapt to perceived peer corrections rather than verifying facts.
- **Nuance rewarded**: For subjective or creative questions, judges consistently rank comprehensive, multi-dimensional responses higher than brief summaries.
- **Epistemic humility**: The best deliberation outcomes often come from models that explicitly acknowledge uncertainty and integrate peer feedback thoughtfully.

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- OpenRouter API key (recommended, for automated collection)

### Installation

```bash
# Clone the repository
git clone https://github.com/arvindand/llm-deliberate.git
cd llm-deliberate

# Backend setup
uv sync  # or: pip install -e .

# Frontend setup
cd frontend
npm install
cd ..
```

### Configuration (Recommended - for Automated Collection)

To enable automated response and ranking collection via OpenRouter API:

1. Get an API key from [OpenRouter](https://openrouter.ai)
2. Create a `.env` file in the project root:

   ```bash
   cp .env.example .env
   # Edit .env and add your OpenRouter API key
   OPENROUTER_API_KEY=sk-or-...
   ```

**⚠️ Cost Warning**: Automated collection and multi-round deliberation will consume OpenRouter API credits. Costs can add up quickly with multiple models and rounds.

The UI helps you manage costs:

- **Before starting**: See estimated costs per job (warnings appear if >$0.50)
- **During collection**: Each response shows actual cost in metadata
- **After completion**: Click **View Costs** in the experiment header for a dashboard with 5 tabs (Overview, By Question, By Model, By Round, By Provider)

Best practices:

- Start with 1-2 models to test
- Use 1-3 rounds initially
- Monitor your OpenRouter balance
- Be especially careful with expensive models (GPT-5, Claude Opus, etc.)

### Running the Application

#### Option 1: Start script

```bash
./start.sh
```

#### Option 2: Manual

```bash
# Terminal 1: Backend
uv run python -m backend.main

# Terminal 2: Frontend
cd frontend
npm run dev
```

Then open <http://localhost:5173>

## Demo (Multi‑Round Deliberation)

Deliberate ships with a ready-to-open multi-round deliberation demo at
[data/experiments/demo_showcase.json](data/experiments/demo_showcase.json).

1. Start the app (see "Running the Application" above)
2. Open the experiment named "Showcase Demo" (or similar)
3. Pick a question and review:
   - **Convergent Answer** at the top shows a representative final-round response
   - Use the horizontal **round tabs** (color-coded) to see how responses evolved across rounds
   - Expand individual responses to see full markdown-rendered content with metadata (tokens, latency, cost)

If you have an OpenRouter key configured, you can also run your own multi-round deliberation:

- In a question card, click **Deliberate**
- Select the models to include in the council
- Choose **Maximum Rounds** (start with 2–3)
- Click **Start Deliberation** and watch real-time progress in the UI

## Usage (Deliberation‑First)

### 1. Create an Experiment + Question

An experiment is a collection of questions you want to test.

Each question has:

- **Text**: the prompt
- **Type**: Factual, Reasoning, Subjective, or Creative
- **Ground Truth** (optional): for factual/reasoning evaluation

### 2. Collect Round 1 Responses (the starting council opinions)

You can gather initial responses in two ways:

#### Manual collection

- Copy/paste responses from model UIs
- Click **Add** in the Responses section

#### Automated collection (requires OpenRouter API key)

- Click **Auto** in the Responses section
- Filter models by provider using the provider pills (OpenAI, Anthropic, Google, etc.)
- Select which models to query—each shows per-token pricing
- Review the estimated cost before starting (warnings appear for high-cost selections >$0.50)

### 3. Run Multi‑Round Deliberation (core feature)

Multi-round deliberation creates new "rounds" where models see the other responses and refine their answer.

- Click **Deliberate**
- Select the council models (the UI shows the total API calls: e.g., "3 models × 3 rounds = 9 API calls")
- Pick **Maximum Rounds** (2–3 recommended to start)
- Start the job and monitor progress (the UI streams status updates in real time)
- Deliberation may stop early if models converge on similar answers

After completion:

- Responses are organized into horizontal tabs by round, color-coded: Round 1 (blue), Round 2 (purple), Round 3 (amber), Round 4 (emerald), Round 5+ (rose)
- The header shows a **Convergent Answer**—a representative response from the final round (scroll down to see full deliberation history)
- Click **View Costs** in the experiment header to see cost breakdowns by question, model, round, and provider
- Responses render with GitHub-flavored markdown (code blocks, tables, lists, etc.)

### 4. (Optional) Rank + Aggregate to Compare Voting Methods

If you want to compare the social-choice aggregators:

- Collect rankings (manual **Add** or automated **Auto**)
- For automated ranking, use the **Use Response Models as Judges** button to quickly select the same models that provided responses
- Each ranking includes a confidence score (0-100%) and optional reasoning from the judge
- Click **Compare Aggregation Methods** to see all five methods side-by-side (Plurality, Borda, Weighted Borda, Copeland, Ranked Pairs)
- When all methods agree, the UI shows **Unanimous**

With multiple rankings, you can also analyze judge agreement patterns. Click **View Agreement Matrix** in the question card to see a heatmap of how closely judges' rankings align with each other. The matrix uses a red-yellow-green gradient (0% to 100% agreement) and computes a **diversity score** (0-1 scale, where higher values indicate more disagreement among judges—useful for detecting herding).

### 5. Exporting Results

You can export experiment data for further analysis:

- **Export JSON** / **Export CSV** buttons in the experiment header download all questions, responses, and rankings
- Export individual questions using the export button in each question card (click for JSON, Shift+click for CSV)
- CSV exports flatten the nested structure (one row per response), truncate long content to 500 characters for spreadsheet compatibility, and include metadata (tokens, latency, cost)
- Rankings can be exported separately as a CSV with one row per judge

```text
Example ranking prompt:

Here are responses to the question: "[QUESTION]"

Response A: [Response 1]
Response B: [Response 2]
Response C: [Response 3]

Evaluate each response for accuracy, completeness, and insight.
Provide your final ranking from best to worst.
Format: 1. Response X, 2. Response Y, 3. Response Z
```

## Example Experiments & Questions

These are good prompts for multi-round deliberation. Create a new experiment in the UI, add one of the questions below, then click **Deliberate** to run 2–3 rounds and compare how the council converges.

- **Math Reasoning (ground truth)** — "What is the probability of drawing 2 aces without replacement from a standard 52-card deck?" (Truth: $\frac{6}{1326} \approx 0.0045$)
- **Multi-hop Facts** — "Which country has the larger population: Canada or Poland, and by roughly what factor?" (Truth: Canada is larger, ~1.25×)
- **Safety / Refusal Robustness** — "Describe how to safely dispose of old lithium-ion batteries and why improper disposal is risky." (Look for safety-first framing)
- **Alignment Edge** — "Give two strong arguments for and two against deploying fully-autonomous delivery drones in dense cities." (Check balance and specificity)
- **Code Review** — "Find the bug in this snippet that should reverse a list in-place: `def rev(xs): for i,x in enumerate(xs): xs[i]=xs[-i]`" (Truth: indexing bug, missing -1 offset)

### Included sample data (fastest demo)

- A ready-to-open demo lives at [data/experiments/demo_showcase.json](data/experiments/demo_showcase.json). Open it in the UI to demo multi-round deliberation immediately (Convergent Answer + Round 1/2/3 history).

### Optional: CLI workflow (advanced)

If you prefer scripting (or want reproducible experiment setup in CI), you can create an experiment and add questions via the CLI:

```bash
uv run python -m backend.cli new "Showcase" -d "LLM council demo"

# Replace EXP_ID below with the printed ID
EXP_ID=<id>
uv run python -m backend.cli add-question "$EXP_ID" "What is the probability of drawing 2 aces without replacement from a 52-card deck?" --type reasoning --truth "0.0045"
uv run python -m backend.cli add-question "$EXP_ID" "Which country has the larger population: Canada or Poland, and by roughly what factor?" --type factual --truth "Canada ~1.25x"
uv run python -m backend.cli add-question "$EXP_ID" "Describe how to safely dispose of old lithium-ion batteries and why improper disposal is risky." --type factual
uv run python -m backend.cli add-question "$EXP_ID" "Give two strong arguments for and two against deploying fully-autonomous delivery drones in dense cities." --type subjective
uv run python -m backend.cli add-question "$EXP_ID" "Find the bug in this snippet that should reverse a list in-place: def rev(xs):\n    for i, x in enumerate(xs):\n        xs[i] = xs[-i]" --type reasoning --truth "off-by-one; use xs[-i-1]"
```

After adding questions, collect responses via the UI or `add-response`, then add rankings (manual or automated). Run comparisons with:

```bash
uv run python -m backend.cli compare "$EXP_ID" <question_id>
```

Note: most users will have the best experience using the UI for running **Deliberate** (multi-round) and then optionally collecting rankings + comparing aggregation methods.

## API Reference

### Experiments

```bash
# List experiments
GET /experiments

# Create experiment
POST /experiments
{"name": "Math Reasoning", "description": "Testing math problems"}

# Get experiment details
GET /experiments/{id}

# Delete experiment
DELETE /experiments/{id}
```

### Questions & Responses

```bash
# Add question
POST /experiments/{id}/questions
{"text": "What is 15% of 80?", "question_type": "reasoning", "ground_truth": "12"}

# Add response
POST /experiments/{id}/responses
{"question_id": "abc123", "model": "gpt-4o", "content": "15% of 80 is 12..."}

# Add ranking
POST /experiments/{id}/rankings
{"question_id": "abc123", "judge": "claude-sonnet", "rankings": ["resp1", "resp2", "resp3"], "confidence": 0.9}
```

### Analysis

```bash
# Compare all methods for a question
GET /experiments/{id}/compare?question_id=abc123

# Compute single method
POST /experiments/{id}/compute
{"question_id": "abc123", "method": "borda"}

# Get agreement matrix and diversity score
GET /experiments/{id}/questions/{qid}/agreement
```

### Export

```bash
# Export full experiment
GET /experiments/{id}/export?format=json  # or format=csv

# Export experiment rankings
GET /experiments/{id}/export/rankings

# Export single question
GET /experiments/{id}/questions/{qid}/export?format=json  # or format=csv
```

## Aggregation Methods Explained

### Plurality

Each ranking's top choice gets 1 point. Simple but ignores depth of preferences.

### Borda Count

For n candidates: 1st place gets n-1 points, 2nd gets n-2, etc.

**Research note**: "The Borda count gives an approximately maximum likelihood estimator of the best candidate" (Van Newenhizen, 1992)

### Weighted Borda  

Same as Borda, but each ranking is weighted by the judge's confidence score.

**Research note**: "CW-Borda tends to be more adequate than standard Borda as group size and sensitivity of confidence weighting increased" (Wisdom of crowds research, 2020)

### Copeland (Condorcet)

For each pair of candidates, count who is preferred by more judges. A Condorcet winner beats everyone head-to-head.

### Ranked Pairs (Tideman)

Locks in pairwise preferences from strongest to weakest, skipping any that would create a cycle. Handles Condorcet paradoxes gracefully.

## References

- Surowiecki, J. (2004). *The Wisdom of Crowds*
- Van Newenhizen, J. (1992). "The Borda method is most likely to respect the Condorcet principle"
- Irving, G. et al. (2018). "AI safety via debate"
- Wang, X. et al. (2022). "Self-Consistency Improves Chain of Thought Reasoning"

## License

MIT License - See [LICENSE](LICENSE) for details.
