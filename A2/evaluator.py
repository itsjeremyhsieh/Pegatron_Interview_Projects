from tqdm import tqdm
from utils.metrics import compute_accuracy


def normalize_to_letter(pred: str) -> str:
    """Extract a single letter (A..J) from model output."""
    if not pred:
        return ""
    import re
    text = str(pred).strip()
    m = re.search(r"\b([A-Ja-j])\b", text)  # standalone text
    if m:
        return m.group(1).upper()
    m = re.search(r"[A-Ja-j]", text)
    return m.group(0).upper() if m else ""


def evaluate_model(model, dataset, max_samples=50):
    y_true_letters, y_pred_letters = [], []

    processed = 0
    pbar = tqdm(total=max_samples)
    for sample in dataset:
        if processed >= max_samples:
            break

        question = sample["question"]
        image = sample["image"]
        options = sample["options"]
        label = sample["label"]

        if not options:
            continue

        # Letters corresponding to options
        letters = [chr(ord('A') + k) for k in range(len(options))]

        pred_raw = model.predict(question, image, options)
        pred_letter = normalize_to_letter(pred_raw)

        if isinstance(label, str) and label.upper() in letters:
            true_letter = label.upper()
        elif isinstance(label, int) and 0 <= label < len(letters):
            true_letter = letters[label]
        else:
            true_letter = ""

        y_true_letters.append(true_letter)
        y_pred_letters.append(pred_letter)

        processed += 1
        pbar.update(1)
    pbar.close()
    correct_count = 0
    print("\nDetail:")
    for i in range(len(y_true_letters)):
        print(
            f"  Sample {i}: True={y_true_letters[i]} | Pred={y_pred_letters[i]} | Match: {y_true_letters[i] == y_pred_letters[i]}")
        if y_true_letters[i] == y_pred_letters[i]:
            correct_count += 1

    acc = compute_accuracy(y_true_letters, y_pred_letters)
    print(f" Accuracy: {acc:.4f}")
    correct_str = f"{correct_count}/{len(y_pred_letters)}"
    return {"accuracy": round(acc, 4), "correct_str": correct_str}
