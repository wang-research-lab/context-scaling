import re
import string
import math

from nltk.metrics.scores import f_measure
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer


def exact_match(gold, pred):
    if not pred:
        return 0
    return 1 if gold.strip() == pred.strip() else 0


def normalize_text(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Copied from the [QuAC](http://quac.ai/) evaluation and script found at
    https://s3.amazonaws.com/my89public/quac/scorer.py"""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def quasi_exact_match(gold, pred):
    if not pred:
        return 0
    return 1 if normalize_text(gold) == normalize_text(pred) else 0


def mc_match(gold, pred):
    """Source: https://arxiv.org/abs/2205.11916"""
    if not pred:
        return 0
    pred = re.findall(r'A|B|C|D|E', pred)
    if pred:
        return 1 if gold == pred[0] else 0
    return 0


def f1_score(gold, pred):
    if not pred:
        return 0
    ret = f_measure(set(normalize_text(gold).split()), set(normalize_text(pred).split()))
    return ret if ret is not None else 0


def bleu_4(gold, pred):
    if not pred:
        return 0
    return sentence_bleu([word_tokenize(gold)], word_tokenize(pred), weights=(0, 0, 0, 1))


def remove_boxed(string):
    """Source: https://github.com/hendrycks/math

    Extract the text within a \\boxed{...} environment.

    Example:
    >>> remove_boxed(\\boxed{\\frac{2}{3}})
    \\frac{2}{3}
    """
    left = "\\boxed{"
    try:
        assert string[: len(left)] == left
        assert string[-1] == "}"
        return string[len(left) : -1]
    except Exception:
        return None


def last_boxed_only_string(string):
    """Source: https://github.com/hendrycks/math

    Extract the last \\boxed{...} or \\fbox{...} element from a string.
    """
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def _fix_fracs(string):
    """Source: https://github.com/hendrycks/math

    Reformat fractions.

    Examples:
    >>> _fix_fracs("\\frac1b")
    \frac{1}{b}
    >>> _fix_fracs("\\frac12")
    \frac{1}{2}
    >>> _fix_fracs("\\frac1{72}")
    \frac{1}{72}
    """
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    """Source: https://github.com/hendrycks/math

    Reformat fractions formatted as a/b to \\frac{a}{b}.

    Example:
    >>> _fix_a_slash_b("2/3")
    \frac{2}{3}
    """
    if len(string.split("/")) != 2:
        return string
    a_str = string.split("/")[0]
    b_str = string.split("/")[1]
    try:
        a = int(a_str)
        b = int(b_str)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def _remove_right_units(string):
    """Source: https://github.com/hendrycks/math

    Remove units (on the right).
    "\\text{ " only ever occurs (at least in the val set) when describing units.
    """
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    """Source: https://github.com/hendrycks/math

    Reformat square roots.

    Example:
    >>> _fix_sqrt("\\sqrt3")
    \sqrt{3}
    """
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    """Source: https://github.com/hendrycks/math

    Apply the reformatting helper functions above.
    """
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc.
    # Even works with \frac1{72} (but not \frac{72}1).
    # Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def get_answer(solution):
    if solution is None:
        return None
    last_boxed = last_boxed_only_string(solution)
    if last_boxed is None:
        return None
    answer = remove_boxed(last_boxed)
    if answer is None:
        return None
    return answer


def is_equiv(str1, str2):
    """Returns (as a float) whether two strings containing math are equivalent up to differences of formatting in
    - units
    - fractions
    - square roots
    - superfluous LaTeX.

    Source: https://github.com/hendrycks/math
    """
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return 1.0
    if str1 is None or str2 is None:
        return 0.0

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        return float(ss1 == ss2)
    except Exception:
        return float(str1 == str2)


def is_equiv_chain_of_thought(gold, pred):
    """Strips the solution first before calling `is_equiv`."""
    return is_equiv(get_answer(gold), get_answer(pred))


def _handle_commas_and_fractions(s):
    if s.count(",") > 0:
        s = s.replace(",", "")
    if s.count("/") == 1:
        numerator, denominator = map(float, s.split("/"))
        return numerator / denominator
    return s

def math_equiv(gold, pred, tol=1e-3):
    try:
        gold = _handle_commas_and_fractions(re.findall(r'\d+(?:[\.,/]\d+)*', gold)[-1])
        pred = _handle_commas_and_fractions(re.findall(r'\d+(?:[\.,/]\d+)*', pred)[-1])
        return 1 if math.isclose(float(gold), float(pred), abs_tol=tol) else 0
    except:
        return 0
    
def rouge2_f1(gold, pred):
    if not pred:
        return 0
    scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    scores = scorer.score(normalize_text(gold), normalize_text(pred))
    return scores['rouge2'].fmeasure

metric_to_metric_fn = {
    "exact_match": exact_match,
    "quasi_exact_match": quasi_exact_match,
    "mc_match": mc_match,
    "f1_score": f1_score,
    "bleu_4": bleu_4,
    "is_equiv_chain_of_thought": is_equiv_chain_of_thought,
    "math_equiv": math_equiv,
    "rouge2_f1": rouge2_f1,
}
