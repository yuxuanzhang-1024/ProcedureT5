from typing import Iterator, List, Optional, Sequence
import textdistance
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm




def full_sentence_accuracy(truth: List[str], pred: List[str]) -> float:
    """
    Calculate the number of exact matches.
    """
    assert len(truth) == len(pred)

    correct_count = sum(int(t == p) for t, p in zip(truth, pred))
    return correct_count / len(truth)

def modified_bleu(truth: List[str], pred: List[str]) -> float:
    """
    Calculates the BLEU score of a translation, with a small modification in order not to penalize sentences
    with less than 4 words.

    Returns:
        value between 0 and 1.
    """
    references = [sentence.split() for sentence in truth]
    candidates = [sentence.split() for sentence in pred]

    # BLEU penalizes sentences with only one word. Even correct translations get a score of zero.
    references = [r + max(0, 4 - len(r)) * [""] for r in references]
    candidates = [c + max(0, 4 - len(c)) * [""] for c in candidates]

    # references must have a larger depth because it supports multiple choices
    refs = [[r] for r in references]
    return corpus_bleu(refs, candidates)  # type: ignore[no-any-return]

def original_bleu(truth: List[str], pred: List[str]) -> float:
    """
    Calculates the BLEU score of a translation, with the original function from nltk.

    Returns:
        value between 0 and 1.
    """
    references = [sentence.split() for sentence in truth]
    candidates = [sentence.split() for sentence in pred]

    # references must have a larger depth because it supports multiple choices
    refs = [[r] for r in references]
    return corpus_bleu(refs, candidates)  # type: ignore[no-any-return]

def levenshtein_similarity(truth: List[str], pred: List[str]) -> float:
    assert len(truth) == len(pred)
    scores = []
    for t, p in tqdm(zip(truth, pred)):
        scores.append(textdistance.levenshtein.normalized_similarity(t, p))
    return scores

def partial_accuracy(truth: List[str], pred: List[str], threshold: float) -> float:
    """
    Calculates the accuracy from the fraction of sentences that have a similarity to the
    ground truth higher than a given threshold.

    For threshold == 1.0, this function is equivalent to full_sentence_accuracy.

    Args:
        truth: ground truth action sequences
        pred: predicted truth action sequences
        threshold: threshold above which to consider it as a partial match, between 0 and 1
    """
    match_count = 0
    assert len(truth) == len(pred)
    for t, p in tqdm(zip(truth, pred)):
        if textdistance.levenshtein.normalized_similarity(t, p) >= threshold:
            match_count += 1
    return match_count / len(truth)