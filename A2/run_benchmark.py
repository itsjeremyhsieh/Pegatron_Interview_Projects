# run_benchmark.py
import argparse
from data_loader import load_mmmu_dataset
from model_interface import BenchmarkModel
from evaluator import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="MMMU Benchmark Pipeline")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., gpt-4o-mini)")
    parser.add_argument("--max_samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--subject", type=str, default="Accounting", help="MMMU subject split (e.g., Accounting, Computer_Science)")
    args = parser.parse_args()

    # Load data
    dataset = load_mmmu_dataset(subject=args.subject)

    # Initialize model
    model = BenchmarkModel(args.model)

    # Run evaluation
    results = evaluate_model(model, dataset, max_samples=args.max_samples)
    print(f"Final results for {args.model}: {results}")

if __name__ == "__main__":
    main()
