import random
from wrapped_datasets.comparison_dataset import ComparisionDataset


# pytest src/wrapped_datasets/test_comparison_dataset.py::test_format_row -s
def test_format_row():
    row = {
        "prompt": "POST: this is a prompt",
        "chosen": "chosen summary",
        "rejected": "rejected summary",
    }

    result = ComparisionDataset.format_row(row, correct_index=0)
    assert (
        result["full_prompt"]
        == "this is a prompt\nA: chosen summary\nB: rejected summary\n"
    )
    assert result["correct_answer"] == "A"

    result = ComparisionDataset.format_row(row, correct_index=1)
    assert (
        result["full_prompt"]
        == "this is a prompt\nA: rejected summary\nB: chosen summary\n"
    )
    assert result["correct_answer"] == "B"

    random.seed(42)
    result = ComparisionDataset.format_row(row)
    assert (
        result["full_prompt"]
        == "this is a prompt\nA: chosen summary\nB: rejected summary\n"
    )
    assert result["correct_answer"] == "A"
