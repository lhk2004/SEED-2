"""GSM8K response processing used by the lm-eval adapter."""


def prepare_gsm8k_response(response):
    """Convert the FIRST box to "#### <answer>", lm-eval's GSM8K answer format.

    Deliberately use string splitting: nested boxes, repeated answers, whitespace,
    and numeric formatting must retain the adapter's existing behavior.
    """
    predicted_answer = None
    if "boxed{" in response:
        predicted_answer = response.split("boxed{")[1].split("}")[0]
        response = response.split("boxed{")[0] + "#### " + predicted_answer
        response = response.replace("$\\", "")
    return response, predicted_answer


def boxed_answer_accuracy(predicted_answer, gold):
    """The adapter's printed Accuracy, distinct from its final lm-eval table."""
    return float(
        predicted_answer is not None and gold.split("### ")[1] == predicted_answer
    )
