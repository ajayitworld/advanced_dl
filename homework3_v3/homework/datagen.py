# used copilot, chatgpt and other online help
def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    """Generate an RFT dataset using a CoT model.

    For each example in the training split, generate `oversample` completions
    (with temperature>0) and keep the first completion whose numeric answer
    matches the ground-truth. Saved format is a JSON list of entries:

        [question: str, answer: float, reasoning: str]

    Args:
        output_json: path to write the resulting json file (e.g. data/rft.json)
        oversample: how many completions to sample per question
        temperature: sampling temperature (>0 to enable diversity)
    """
    import json
    from pathlib import Path
    from tqdm import tqdm

    from .cot import load as load_cot
    from .data import Dataset, is_answer_valid

    model = load_cot(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    model.model.eval()

    trainset = Dataset("train")

    results = []

    for q, correct in tqdm(trainset, desc="Generating rollouts"):
        # Format the prompt using the CoT formatting (includes examples)
        formatted = model.format_prompt(q)

        # Generate multiple diverse completions
        generations = model.batched_generate([formatted], num_return_sequences=oversample, temperature=temperature)
        # batched_generate returns list[list[str]] when num_return_sequences provided
        if isinstance(generations, list) and len(generations) > 0 and isinstance(generations[0], list):
            gens = generations[0]
        else:
            gens = generations if isinstance(generations, list) else [generations]

        # Find the first generation that yields a correct numeric answer
        selected = None
        for g in gens:
            try:
                ans = model.parse_answer(g)
            except Exception:
                ans = float("nan")
            if is_answer_valid(ans, correct, relative_tolerance=0.05):
                selected = g
                break

        if selected is not None:
            results.append([q, float(correct), selected])

    # Write results
    output_json = 'data/rft.json'
    outp = Path(output_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} examples to {output_json}")

    return results


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
