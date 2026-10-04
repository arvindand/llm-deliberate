

# LLM Deliberate

**An experimentation tool for exploring multi-model LLM deliberation and aggregation methods.**

> Inspired by [Andrej Karpathy's llm-council](https://github.com/karpathy/llm-council)

![Deliberate UI](docs/screenshot-deliberation.png)

## ⚠️ Maintenance Disclaimer

**This is an experimental personal research project.** Like Karpathy's original llm-council, it is primarily an exploration of LLM deliberation methods. It is maintained on a best-effort basis and the APIs and data format may evolve between versions.

Feel free to fork and modify it for your own research needs!

## How Deliberate Differs from llm-council

While inspired by Karpathy's llm-council, Deliberate takes a different approach:

| Feature | llm-council | Deliberate |
| ------- | ----------- | ---------- |
| **Purpose** | Chat interface with synthesized final answer | Experimentation tool for studying aggregation methods |
| **Output** | Single "Chairman" synthesized response | Side-by-side comparison of **8** voting algorithms |
| **Deliberation** | Single round + review | Multi-round iterative refinement |
| **Analysis** | Rankings displayed | Agreement matrices, cost dashboards, export to CSV/JSON |
| **Data Model** | Conversation threads | Structured experiments with questions |
| **Manual Entry** | Not supported | Full support for manual data collection |
| **Self-Consistency** | Not supported | Query same model N times (Wang et al. 2022) |
| **Debate Format** | Not supported | Adversarial deliberation API with judge (Irving et al. 2018) |
| **Dark Mode** | Not supported | Warm brown theme with toggle |

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

For the aggregation layer, Deliberate implements eight algorithms from social choice theory:

- **Plurality** — Simple first-place vote counting
- **Borda Count** — Positional voting with points for each rank
- **Weighted Borda** — Borda weighted by judge confidence scores
- **Copeland (Condorcet)** — Pairwise comparison winner
- **Ranked Pairs (Tideman)** — Handles voting cycles gracefully
- **Schulze Method** — Strongest path Condorcet completion
- **STV/Instant Runoff** — Iterative elimination voting
- **Approval Voting** — Top-N approval threshold counting

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

> **New here?** See [QUICKSTART.md](QUICKSTART.md) for a hands-on walkthrough of all features.

### Prerequisites

- Python 3.10+
- Node.js 20.19+ (or 22.12+)
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

## Quick Feature Test

After starting the app, test these features:

| Feature | How to Test |
|---------|-------------|
| Dark Mode | Click moon icon in header |
| 8 Aggregation Methods | Rankings → Compare → All 8 appear |
| Evolution View | Multi-round question → Toggle "Evolution" |
| Agreement Heatmap | Rankings → "View Agreement" button |
| Chairman Synthesis | Rankings → "Synthesize Final Answer" button |
| Debate API | Open `http://localhost:8000/docs` → `POST /experiments/{id}/automate/debate` |

## Testing

```bash
# Backend
pytest

# Frontend
cd frontend
npm test
npm run build
```

## Demo (Multi‑Round Deliberation)

A multi-round deliberation demo is available at
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

### 2. Choose Your Response Collection Mode

> **Important:** Auto collection and Deliberation are **mutually exclusive** per question. Choose one approach:

#### Option A: One-Shot Collection (Auto)

Collect a single response from each model with no iteration:

- Click **Auto** in the Responses section
- Filter models by provider using the provider pills (OpenAI, Anthropic, Google, etc.)
- Select which models to query—each shows per-token pricing
- Review the estimated cost before starting (warnings appear for high-cost selections >$0.50)

Best for: Comparing initial model outputs, collecting responses for ranking/aggregation

#### Option B: Multi-Round Deliberation (core feature)

Models see each other's responses and refine their answers over multiple rounds:

- Click **Deliberate**
- Select the council models (the UI shows the total API calls: e.g., "3 models × 3 rounds = 9 API calls")
- Pick **Maximum Rounds** (2–3 recommended to start)
- Start the job and monitor progress (the UI streams status updates in real time)
- Deliberation may stop early if models converge on similar answers

Best for: Studying consensus formation, comparing how models update their reasoning

After deliberation completes:

- Responses are organized into horizontal tabs by round, color-coded: Round 1 (blue), Round 2 (purple), Round 3 (amber), Round 4 (emerald), Round 5+ (rose)
- The header shows a **Convergent Answer**—a representative response from the final round (scroll down to see full deliberation history)
- Click **View Costs** in the experiment header to see cost breakdowns by question, model, round, and provider

#### Option C: Manual Collection

- Copy/paste responses from model UIs
- Click **Add** in the Responses section

All response modes render with GitHub-flavored markdown (code blocks, tables, lists, etc.)

### 3. (Optional) Rank + Aggregate to Compare Voting Methods

If you want to compare the social-choice aggregators:

- Collect rankings (manual **Add** or automated **Auto**)
- For automated ranking, use the **Use Response Models as Judges** button to quickly select the same models that provided responses
- Each ranking includes a confidence score (0-100%) and optional reasoning from the judge
- Click **Compare Aggregation Methods** to see all eight methods side-by-side
- When all methods agree, the UI shows **Unanimous**

With multiple rankings, you can also analyze judge agreement patterns. Click **View Agreement Matrix** in the question card to see a heatmap of how closely judges' rankings align with each other. The matrix uses a red-yellow-green gradient (0% to 100% agreement) and computes a **diversity score** (0-1 scale, where higher values indicate more disagreement among judges—useful for detecting herding).

## Advanced Features

### Self-Consistency Mode (API Only)

Query the same model N times with temperature=0.8 to sample diverse reasoning paths. This feature is available via the REST API but not exposed in the UI.

```bash
POST /experiments/{id}/automate/self-consistency
{
  "question_id": "...",
  "model": "anthropic/claude-sonnet-4",
  "num_samples": 5
}
```

Based on: Wang et al., 2022 - "Self-Consistency Improves Chain of Thought Reasoning"

### Debate Format (API Only)

Run adversarial deliberation where two models argue and a judge picks the winner. This
workflow is currently available through the REST API rather than the UI.

Use the interactive API documentation at <http://localhost:8000/docs>, or call:

```text
POST /experiments/{id}/automate/debate
{
  "question_id": "...",
  "debaters": ["model-a", "model-b"],
  "judge_model": "model-c",
  "num_rounds": 3
}
```

Poll the returned job ID through the automation status endpoint to retrieve the
arguments and judge verdict.

Based on: Irving et al., 2018 - "AI safety via debate"

### Evolution Visualization

Track how responses change across deliberation rounds:

- Run multi-round deliberation (2+ rounds)
- Toggle **"Evolution"** view in the Responses section
- See convergence chart and per-model timeline
- Expand rounds to see word-level diffs

### Chairman Synthesis

Generate a final synthesized answer from top-ranked responses:

- Collect rankings for a question
- Click **"Synthesize Final Answer"** button
- Select a chairman model
- View the markdown-rendered synthesis; it is saved with the experiment and remains
  available after reloading the page

### Dark Mode

Toggle dark mode via the moon/sun icon in the header. Uses warm brown tones (#2a2318) instead of pure black. Preference saved to localStorage.

## Exporting Results

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
# Or, after installation:
uv run deliberate new "Showcase" -d "LLM council demo"

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

`compute` and `compare` scores are keyed by **response ID**. `response_labels`
maps each ID to its model, round, and ID so repeated samples stay distinct.
Every method returns `winner_ids`, a nullable singular winner, and `status`.
`compare.winner` is a response ID; `compute.winner` retains its response object
for a unique winner and is `null` otherwise. `compute.raw_scores` remains an
alias of the ID-keyed scores. Clients that previously used model-keyed `scores`
or model names in `compare.winner` must use `response_labels` for display.
Saved experiment files require no migration.

A score tie returns all top response IDs with `status: "tie"`. An IRV
elimination tie instead returns `status: "unresolved"`, no elected winner IDs,
and `details.remaining_ids`. `unanimous` requires every method to elect the same
single response; agreement on a model name or an unresolved count does not qualify.
The score-only Python helpers `get_winner` and `get_winners` return a unique
top ID (or `None` for a tie) and every top ID, respectively. They cannot infer
whether an IRV count stopped unresolved from scores alone. Use
`aggregate_rankings` for count status and elected winner IDs. `method_agreement`
now maps methods to winner-ID lists (empty for an unresolved count).

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

The winner is an undefeated root of the locked graph that appears on at least
one ballot. Scores indicate eligibility (`1` ranked and undefeated, `0` defeated
or unranked), rather than counting outgoing victories. Responses added after
judging cannot win without being ranked. Equal margins use winner-ID then
loser-ID order; tied-strength cycles can depend on that explicit tiebreak.
Multiple eligible undefeated roots are reported as tied.

### Schulze Method

Computes the strongest path between all candidate pairs using the Floyd-Warshall algorithm. Handles Condorcet cycles elegantly by finding the candidate with the strongest indirect comparisons.

**Research note**: "A New Monotonic, Clone-Independent, Reversal Symmetric, and Condorcet-Consistent Single-Winner Election Method" (Markus Schulze, 2011)

### STV / Instant Runoff

Eliminates the lowest-ranked candidate each round, transferring votes to voters' next choices. Used in Australian federal elections and many other jurisdictions worldwide.

This implementation removes zero-vote candidates together, but stops if a
positive-vote minimum is tied. Choosing an elimination order or removing all
tied contenders can change the result, so the remaining candidates are reported
as unresolved rather than elected. Its scores describe outcome eligibility,
not first-place vote totals or a complete ranking.

### Approval Voting

Candidates in the top N positions (default: 2) are "approved" by that judge. Counts total approvals across all judges.

**Research note**: "Approval Voting" - Brams & Fishburn (1983)

## References

- Surowiecki, J. (2004). *The Wisdom of Crowds*
- Van Newenhizen, J. (1992). "The Borda method is most likely to respect the Condorcet principle"
- Irving, G. et al. (2018). "AI safety via debate"
- Wang, X. et al. (2022). "Self-Consistency Improves Chain of Thought Reasoning"
- Schulze, M. (2011). "A New Monotonic, Clone-Independent, Reversal Symmetric, and Condorcet-Consistent Single-Winner Election Method"
- Brams, S. J. & Fishburn, P. C. (1983). "Approval Voting"

## License

MIT License - See [LICENSE](LICENSE) for details.
