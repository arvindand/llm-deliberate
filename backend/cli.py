#!/usr/bin/env python3
"""
CLI tool for LLM Deliberate experiments.

Usage:
    python -m backend.cli new "My Experiment"
    python -m backend.cli add-question <exp_id> "What is 2+2?" --type reasoning --truth "4"
    python -m backend.cli add-response <exp_id> <question_id> gpt-4o "The answer is 4"
    python -m backend.cli add-ranking <exp_id> <question_id> claude-sonnet resp1,resp2,resp3
    python -m backend.cli compare <exp_id> <question_id>
    python -m backend.cli list
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models import Experiment, Question, Response, Ranking, QuestionType
from backend.aggregation import (
    borda_count, weighted_borda, copeland_score, plurality, ranked_pairs,
    get_winner, method_agreement, diversity_score
)

# Constants
EXPERIMENT_ID_HELP = "Experiment ID"
QUESTION_ID_HELP = "Question ID"

DATA_DIR = Path(__file__).parent.parent / "data" / "experiments"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_experiment(exp_id: str) -> Experiment:
    path = DATA_DIR / f"{exp_id}.json"
    if not path.exists():
        print(f"❌ Experiment '{exp_id}' not found")
        sys.exit(1)
    with open(path) as f:
        return Experiment(**json.load(f))


def save_experiment(exp: Experiment):
    path = DATA_DIR / f"{exp.id}.json"
    with open(path, "w") as f:
        json.dump(exp.model_dump(), f, indent=2, default=str)


def cmd_new(args):
    """Create a new experiment."""
    exp = Experiment(name=args.name, description=args.description)
    save_experiment(exp)
    print(f"✅ Created experiment: {exp.id}")
    print(f"   Name: {exp.name}")


def cmd_list(args):
    """List all experiments."""
    experiments = list(DATA_DIR.glob("*.json"))
    if not experiments:
        print("No experiments found.")
        return
    
    print(f"\n{'ID':<10} {'Name':<30} {'Questions':<10}")
    print("-" * 55)
    for path in experiments:
        with open(path) as f:
            data = json.load(f)
        print(f"{data['id']:<10} {data['name'][:28]:<30} {len(data.get('questions', [])):<10}")


def cmd_show(args):
    """Show experiment details."""
    exp = load_experiment(args.exp_id)
    
    print(f"\n📋 Experiment: {exp.name}")
    print(f"   ID: {exp.id}")
    print(f"   Description: {exp.description or '(none)'}")
    print(f"   Created: {exp.created_at}")
    print(f"\n   Questions ({len(exp.questions)}):")
    
    for q in exp.questions:
        print(f"\n   [{q.id}] {q.text[:60]}{'...' if len(q.text) > 60 else ''}")
        print(f"       Type: {q.question_type} | Responses: {len(q.responses)} | Rankings: {len(q.rankings)}")
        if q.ground_truth:
            print(f"       Ground truth: {q.ground_truth}")
        
        for r in q.responses:
            print(f"       - {r.model} ({r.id}): {r.content[:40]}...")


def cmd_add_question(args):
    """Add a question to an experiment."""
    exp = load_experiment(args.exp_id)
    
    q = Question(
        text=args.text,
        question_type=QuestionType(args.type),
        ground_truth=args.truth
    )
    exp.questions.append(q)
    save_experiment(exp)
    
    print(f"✅ Added question: {q.id}")


def cmd_add_response(args):
    """Add a model response to a question."""
    exp = load_experiment(args.exp_id)
    
    # Find question
    question = None
    for q in exp.questions:
        if q.id == args.question_id:
            question = q
            break
    
    if not question:
        print(f"❌ Question '{args.question_id}' not found")
        sys.exit(1)
    
    response = Response(model=args.model, content=args.content)
    question.responses.append(response)
    save_experiment(exp)
    
    print(f"✅ Added response: {response.id} ({args.model})")


def cmd_add_ranking(args):
    """Add a ranking to a question."""
    exp = load_experiment(args.exp_id)
    
    # Find question
    question = None
    for q in exp.questions:
        if q.id == args.question_id:
            question = q
            break
    
    if not question:
        print(f"❌ Question '{args.question_id}' not found")
        sys.exit(1)
    
    rankings = [r.strip() for r in args.rankings.split(",")]
    
    ranking = Ranking(
        judge=args.judge,
        rankings=rankings,
        confidence=args.confidence
    )
    question.rankings.append(ranking)
    save_experiment(exp)
    
    print(f"✅ Added ranking from {args.judge}: {' > '.join(rankings)}")


def _find_question(exp: Experiment, question_id: str) -> Question | None:
    """Find a question by ID in an experiment."""
    for q in exp.questions:
        if q.id == question_id:
            return q
    return None


def _print_method_scores(name: str, scores: dict[str, float], 
                        id_to_model: dict[str, str], winner_id: str) -> None:
    """Print scores for a single aggregation method."""
    print(f"   {name}:")
    for cid, score in sorted(scores.items(), key=lambda x: -x[1]):
        model = id_to_model.get(cid, cid)
        bar = "█" * int(score * 2) if score > 0 else ""
        marker = " 🏆" if cid == winner_id else ""
        print(f"      {model:<15} {score:>5.1f} {bar}{marker}")
    print()


def _print_unanimity(winners: list[str], methods: dict) -> None:
    """Print unanimity or disagreement status."""
    if len(set(winners)) == 1:
        print(f"   ✅ UNANIMOUS: All methods agree on {winners[0]}")
    else:
        print("   ⚠️  SPLIT: Methods disagree")
        for name, winner in zip(methods.keys(), winners):
            print(f"      {name}: {winner}")


def cmd_compare(args):
    """Compare aggregation methods for a question."""
    exp = load_experiment(args.exp_id)
    
    question = _find_question(exp, args.question_id)
    if not question:
        print(f"❌ Question '{args.question_id}' not found")
        sys.exit(1)
    
    if not question.rankings:
        print("❌ No rankings available")
        sys.exit(1)
    
    candidates = [r.id for r in question.responses]
    id_to_model = {r.id: r.model for r in question.responses}
    
    print(f"\n📊 Comparison for: {question.text[:60]}...")
    print(f"   Responses: {len(question.responses)} | Rankings: {len(question.rankings)}")
    print()
    
    methods = {
        "Plurality": plurality,
        "Borda Count": borda_count,
        "Weighted Borda": weighted_borda,
        "Copeland": copeland_score,
        "Ranked Pairs": ranked_pairs
    }
    
    winners = []
    for name, method in methods.items():
        scores = method(question.rankings, candidates)
        winner_id = get_winner(scores)
        winner_model = id_to_model.get(winner_id, winner_id)
        winners.append(winner_model)
        _print_method_scores(name, scores, id_to_model, winner_id)
    
    _print_unanimity(winners, methods)
    
    # Diversity score
    div = diversity_score(question.rankings, candidates)
    print(f"\n   Diversity score: {div:.2f} (0=agreement, 1=disagreement)")
    
    if question.ground_truth:
        print(f"\n   Ground truth: {question.ground_truth}")


def _collect_responses(question: Question) -> None:
    """Collect responses interactively from user."""
    print("\n--- Adding Responses ---")
    print("(Enter model name, then paste response. Empty model name to finish.)\n")
    
    while True:
        model = input("Model name (or empty to finish): ").strip()
        if not model:
            break
        
        print("Paste response (end with empty line):")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        
        content = "\n".join(lines)
        if content:
            response = Response(model=model, content=content)
            question.responses.append(response)
            print(f"✅ Added response {response.id} from {model}\n")


def _collect_rankings(question: Question) -> None:
    """Collect rankings interactively from user."""
    print("\n--- Adding Rankings ---")
    print("Available responses:")
    for r in question.responses:
        print(f"  {r.id}: {r.model} - {r.content[:50]}...")
    
    print("\n(Enter judge model, then ranking as comma-separated IDs. Empty to finish.)\n")
    
    while True:
        judge = input("Judge model (or empty to finish): ").strip()
        if not judge:
            break
        
        ranking_str = input("Ranking (best to worst, comma-separated IDs): ").strip()
        rankings = [r.strip() for r in ranking_str.split(",") if r.strip()]
        
        if rankings:
            conf = input("Confidence 0-1 (default 1.0): ").strip()
            confidence = float(conf) if conf else 1.0
            
            ranking = Ranking(judge=judge, rankings=rankings, confidence=confidence)
            question.rankings.append(ranking)
            print(f"✅ Added ranking from {judge}\n")


def cmd_interactive(args):
    """Interactive mode for adding responses and rankings."""
    exp = load_experiment(args.exp_id)
    
    question = _find_question(exp, args.question_id)
    if not question:
        print(f"❌ Question '{args.question_id}' not found")
        sys.exit(1)
    
    print(f"\n📝 Question: {question.text}")
    print(f"   Type: {question.question_type}")
    if question.ground_truth:
        print(f"   Ground truth: {question.ground_truth}")
    
    _collect_responses(question)
    save_experiment(exp)
    
    if len(question.responses) < 2:
        print("Need at least 2 responses to add rankings.")
        return
    
    _collect_rankings(question)
    save_experiment(exp)
    print("\n✅ Saved all data. Run 'compare' to see results.")


def main():
    parser = argparse.ArgumentParser(
        description="LLM Deliberate CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # new
    p = subparsers.add_parser("new", help="Create new experiment")
    p.add_argument("name", help="Experiment name")
    p.add_argument("--description", "-d", help="Description")
    p.set_defaults(func=cmd_new)
    
    # list
    p = subparsers.add_parser("list", help="List experiments")
    p.set_defaults(func=cmd_list)
    
    # show
    p = subparsers.add_parser("show", help="Show experiment details")
    p.add_argument("exp_id", help=EXPERIMENT_ID_HELP)
    p.set_defaults(func=cmd_show)
    
    # add-question
    p = subparsers.add_parser("add-question", help="Add question")
    p.add_argument("exp_id", help=EXPERIMENT_ID_HELP)
    p.add_argument("text", help="Question text")
    p.add_argument("--type", "-t", default="reasoning", 
                   choices=["factual", "reasoning", "subjective", "creative"])
    p.add_argument("--truth", help="Ground truth answer")
    p.set_defaults(func=cmd_add_question)
    
    # add-response
    p = subparsers.add_parser("add-response", help="Add model response")
    p.add_argument("exp_id", help=EXPERIMENT_ID_HELP)
    p.add_argument("question_id", help=QUESTION_ID_HELP)
    p.add_argument("model", help="Model name (e.g., gpt-4o)")
    p.add_argument("content", help="Response content")
    p.set_defaults(func=cmd_add_response)
    
    # add-ranking
    p = subparsers.add_parser("add-ranking", help="Add ranking")
    p.add_argument("exp_id", help=EXPERIMENT_ID_HELP)
    p.add_argument("question_id", help=QUESTION_ID_HELP)
    p.add_argument("judge", help="Judge model name")
    p.add_argument("rankings", help="Comma-separated response IDs (best to worst)")
    p.add_argument("--confidence", "-c", type=float, default=1.0)
    p.set_defaults(func=cmd_add_ranking)
    
    # compare
    p = subparsers.add_parser("compare", help="Compare aggregation methods")
    p.add_argument("exp_id", help=EXPERIMENT_ID_HELP)
    p.add_argument("question_id", help=QUESTION_ID_HELP)
    p.set_defaults(func=cmd_compare)
    
    # interactive
    p = subparsers.add_parser("interactive", help="Interactive mode")
    p.add_argument("exp_id", help=EXPERIMENT_ID_HELP)
    p.add_argument("question_id", help=QUESTION_ID_HELP)
    p.set_defaults(func=cmd_interactive)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
