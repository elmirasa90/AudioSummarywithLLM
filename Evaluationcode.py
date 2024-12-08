import pandas as pd
from rouge_score import rouge_scorer
from bert_score import score
from nltk.translate.meteor_score import meteor_score
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

def preprocess_text(text):
    """Preprocess and validate text."""
    if isinstance(text, float):  # Handle numerical values
        text = str(text)
    if not isinstance(text, str) or text.strip() in ["...", "", "nan", "None"]:
        return None
    return text.strip()

def compute_metrics(model_summary, reference_summary, scorer):
    """Compute ROUGE, BERTScore, and METEOR metrics."""
    try:
        # Validate and preprocess inputs
        model_summary = preprocess_text(model_summary)
        reference_summary = preprocess_text(reference_summary)

        if not model_summary or not reference_summary:
            raise ValueError("Invalid or empty summary inputs.")

        # Tokenize for METEOR
        model_tokens = model_summary.split()
        reference_tokens = reference_summary.split()

        # Compute ROUGE
        rouge_scores = scorer.score(model_summary, reference_summary)
        rouge_1 = rouge_scores["rouge1"].fmeasure
        rouge_2 = rouge_scores["rouge2"].fmeasure
        rouge_l = rouge_scores["rougeL"].fmeasure

        # Compute BERTScore
        P, R, F1 = score(
            [model_summary], [reference_summary], lang="en", verbose=False
        )
        bert_f1 = F1.mean().item()

        # Compute METEOR
        meteor = meteor_score([reference_tokens], model_tokens)

        return {
            "ROUGE-1": rouge_1,
            "ROUGE-2": rouge_2,
            "ROUGE-L": rouge_l,
            "BERTScore": bert_f1,
            "METEOR": meteor,
        }
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return None

def main():
    # Path to the input file
    file_path = "D:\Speech Summerization\Final_Experiments\combined_summaries_1.xlsx"


    data = pd.read_excel(file_path)

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Initialize accumulators
    results = []
    rouge_1_scores, rouge_2_scores, rouge_l_scores, bert_scores, meteor_scores = [], [], [], [], []

    print("Starting evaluation...\n")

    # Iterate over each row to calculate metrics
    for idx, row in data.iterrows():
        print(f"Processing row {idx}...")

        model_summary = row.get("Summary by model", None)
        reference_summary = row.get("Reference Summary", None)

        metrics = compute_metrics(model_summary, reference_summary, scorer)
        if metrics:
            print("  Metrics computed successfully:")
            for key, value in metrics.items():
                print(f"    {key}: {value:.4f}")

            # Append metrics to accumulators
            rouge_1_scores.append(metrics["ROUGE-1"])
            rouge_2_scores.append(metrics["ROUGE-2"])
            rouge_l_scores.append(metrics["ROUGE-L"])
            bert_scores.append(metrics["BERTScore"])
            meteor_scores.append(metrics["METEOR"])
        else:
            print("  Metrics computation failed.")

    # Calculate mean scores if there are valid results
    if rouge_1_scores:
        print("\nFinal Evaluation Results (Mean Scores):")
        print(f"  Mean ROUGE-1: {sum(rouge_1_scores) / len(rouge_1_scores):.4f}")
        print(f"  Mean ROUGE-2: {sum(rouge_2_scores) / len(rouge_2_scores):.4f}")
        print(f"  Mean ROUGE-L: {sum(rouge_l_scores) / len(rouge_l_scores):.4f}")
        print(f"  Mean BERTScore: {sum(bert_scores) / len(bert_scores):.4f}")
        print(f"  Mean METEOR: {sum(meteor_scores) / len(meteor_scores):.4f}")
    else:
        print("No valid metrics to compute mean scores.")

if __name__ == "__main__":
    main()
