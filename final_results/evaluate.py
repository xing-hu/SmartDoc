from sumeval.metrics.rouge import RougeCalculator
import nltk
import numpy as np


def eval_leclair(ref, hyp):
    with open(ref, "r") as f:
        refs = [line.strip() for line in f.readlines()]
    with open(hyp, "r") as f:
        hypotheses = [line.strip() for line in f.readlines()]
    refs = [[r.split()] for r in refs]
    preds = [h.split() for h in hypotheses]
    # Adapted from https://github.com/mcmillco/funcom/blob/master/bleu.py
    Ba = nltk.translate.bleu_score.corpus_bleu(refs, preds)
    B1 = nltk.translate.bleu_score.corpus_bleu(refs, preds, weights=(1, 0, 0, 0))
    B2 = nltk.translate.bleu_score.corpus_bleu(refs, preds, weights=(0, 1, 0, 0))
    B3 = nltk.translate.bleu_score.corpus_bleu(refs, preds, weights=(0, 0, 1, 0))
    B4 = nltk.translate.bleu_score.corpus_bleu(refs, preds, weights=(0, 0, 0, 1))

    Ba = round(Ba * 100, 2)
    B1 = round(B1 * 100, 2)
    B2 = round(B2 * 100, 2)
    B3 = round(B3 * 100, 2)
    B4 = round(B4 * 100, 2)

    ret = ''
    ret += ('BLEU for %s functions\n' % (len(preds)))
    ret += ('Ba %s\n' % (Ba))
    ret += ('B1 %s\n' % (B1))
    ret += ('B2 %s\n' % (B2))
    ret += ('B3 %s\n' % (B3))
    ret += ('B4 %s\n' % (B4))
    #print(ret)
    return [Ba, B1, B2, B3, B4]


def _rouge(rouge, model, reference):
    """
    calculate rouge socore
    :param rouge: sumeval instance
    :param model: model prediction, list
    :param reference: reference
    :param type: rouge1, rouge2, rougel
    :return: rouge 1 score
    """
    scores = rouge.rouge_l(summary=model, references=reference)

    return scores


def eval_rouge(ref, hyp):
    glines = []
    with open(ref, 'r') as gfile:
        lines = gfile.readlines()
        for line in lines:
            glines.append(line.strip())

    olines = []
    with open(hyp, 'r') as file:
        lines = file.readlines()
        for line in lines:
            olines.append(line.strip())
    rouge = RougeCalculator(stopwords=True, lang="en")

    rougel_scores = [_rouge(rouge, model, [reference]) for model, reference in zip(olines, glines)]
    rougel_score = sum(rougel_scores) / len(rougel_scores)
    #print("ROUGe: ", rougel_score)
    return rougel_score


# Evaluate RQ1
for app in ["attendgru", "ast-attendgru", "re2com", "smartdoc"]:
    print("Results for approach %s:" % app)
    B = eval_leclair("./RQ1/ref.txt", "./RQ1/%s.out" % app)
    R = eval_rouge("./RQ1/ref.txt", "./RQ1/%s.out" % app)
    print(B + [R])
    print("*******************")

# Evaluate RQ2
for app in ["transformer", "trans_pointer", "smartdoc"]:
    print("Results for approach %s:" % app)
    B = eval_leclair("./RQ2/ref.txt", "./RQ2/%s.out" % app)
    R = eval_rouge("./RQ2/ref.txt", "./RQ2/%s.out" % app)
    print(B + [R])
    print("*******************")

# Evaluate Cross Project
for app in ["attendgru", "ast-attendgru", "re2com", "smartdoc"]:
    print("Results for approach %s:" % app)
    scores = []
    for fold in ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]:

        B = eval_leclair("./cross_project/%s/ref.txt" % fold, "./cross_project/%s/%s.out" % (fold, app))
        r = eval_rouge("./cross_project/%s/ref.txt" % fold, "./cross_project/%s/%s.out" % (fold, app))
        score = B + [r]
        scores.append(score)

    res = np.array(scores)
    res = res.mean(axis=0)
    print(res)
    print("*********************")