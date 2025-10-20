from datasets import load_dataset, Image as HFImage


def load_mmmu_dataset(split="validation", subject: str = "Accounting"):
    """
    Load and preprocess the MMMU dataset.
    """
    print(f"Loading MMMU dataset split: {split}, subject: {subject}")
    dataset = load_dataset("MMMU/MMMU", subject, split=split)
    try:
        dataset = dataset.cast_column("image_1", HFImage())
    except Exception as _e:
        print(
            f"Warning: could not cast 'image_1' column to PIL: {_e}.")

    def preprocess(data):
        # Options are stored as a string that looks like a Python list
        options_str = data.get("options", "[]")

        # Parse the string representation of the list
        import ast
        try:
            options_list = ast.literal_eval(options_str)
            if not isinstance(options_list, list):
                options_list = []
        except (ValueError, SyntaxError):
            print(f"Warning: Failed to parse options: {options_str}")
            options_list = []
        image = data.get("image_1")

        return {
            "id": data.get("id"),
            "question": data.get("question"),
            "image": image,
            "options": options_list,
            "label": data.get("answer")
        }

    dataset = dataset.map(preprocess)
    return dataset
