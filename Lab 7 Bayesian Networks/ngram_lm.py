"""
CS F407 Lab: Bayesian Networks and Autoregressive Language Models.

First-order  (bigram)  model: P(X_t | X_{t-1})           BN: X1 -> X2 -> X3 -> ...
Second-order (trigram) model: P(X_t | X_{t-2}, X_{t-1})  BN: X_{t-2} -> X_t <- X_{t-1}

Both are estimated by counting (maximum likelihood) with plain Python data
structures: no ML library, no pretrained model.

Usage:
    python3 ngram_lm.py        # runs every part of the lab and writes the
                               # generated sentences to generated_*.txt
"""

import random
from collections import Counter, defaultdict

START, END = "<START>", "<END>"

CORPUS = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]

MAX_LEN = 30  # safety cap on generated sentence length (tokens, excluding markers)


def tokenise(sentence):
    return sentence.lower().split()


# ---------------------------------------------------------------------------
# First-order model
# ---------------------------------------------------------------------------
class FirstOrderLM:
    """P(X_t | X_{t-1}) estimated from bigram counts."""

    order = 1

    def __init__(self, sentences):
        # counts[prev][next] = C(prev, next): transition counts live here.
        self.counts = defaultdict(Counter)
        for tokens in sentences:
            seq = [START] + tokens + [END]
            for prev, nxt in zip(seq, seq[1:]):
                self.counts[prev][nxt] += 1
        # P(next | prev) = C(prev, next) / sum_k C(prev, k)
        self.probs = {}
        for prev, nexts in self.counts.items():
            total = sum(nexts.values())
            self.probs[prev] = {w: c / total for w, c in nexts.items()}

    def context(self, history):
        return history[-1]

    def distribution(self, history):
        """P(X_t | context). Empty dict when the context was never observed."""
        return self.probs.get(self.context(history), {})

    def predict(self, history):
        dist = self.distribution(history)
        if not dist:
            return None
        # argmax; ties broken alphabetically so greedy output is reproducible
        return max(sorted(dist), key=lambda w: dist[w])

    def sample(self, history, rng):
        dist = self.distribution(history)
        if not dist:
            return None
        words = list(dist)
        return rng.choices(words, weights=[dist[w] for w in words], k=1)[0]

    def start_history(self):
        return [START] * self.order

    def generate(self, rng=None, greedy=False):
        """Returns (tokens, stopped_reason)."""
        history = self.start_history()
        out = []
        while len(out) < MAX_LEN:
            nxt = self.predict(history) if greedy else self.sample(history, rng)
            if nxt is None:
                return out, "unseen context"
            if nxt == END:
                return out, "END"
            out.append(nxt)
            history.append(nxt)
        return out, f"hit MAX_LEN={MAX_LEN}"

    def sentence_probability(self, tokens):
        history = self.start_history()
        p = 1.0
        for w in tokens + [END]:
            p *= self.distribution(history).get(w, 0.0)
            history.append(w)
        return p


# ---------------------------------------------------------------------------
# Second-order model
# ---------------------------------------------------------------------------
class SecondOrderLM(FirstOrderLM):
    """P(X_t | X_{t-2}, X_{t-1}) estimated from trigram counts.

    Every sentence is padded with two <START> tokens, so
    P(X1) = P(X1 | <START>, <START>) and P(X2 | X1) = P(X2 | <START>, X1).
    """

    order = 2

    def __init__(self, sentences):
        self.counts = defaultdict(Counter)
        for tokens in sentences:
            seq = [START, START] + tokens + [END]
            for a, b, c in zip(seq, seq[1:], seq[2:]):
                self.counts[(a, b)][c] += 1
        self.probs = {}
        for ctx, nexts in self.counts.items():
            total = sum(nexts.values())
            self.probs[ctx] = {w: n / total for w, n in nexts.items()}

    def context(self, history):
        return (history[-2], history[-1])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def banner(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


def fmt_dist(dist):
    return ", ".join(f"{w}: {p:.4f}" for w, p in sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])))


def normalisation_test(model):
    """Every conditional distribution must sum to 1."""
    worst = 0.0
    for ctx, dist in model.probs.items():
        worst = max(worst, abs(sum(dist.values()) - 1.0))
    return worst


def vocabulary(sentences):
    return sorted({w for s in sentences for w in s})


def main():
    rng = random.Random(407)
    data = [tokenise(s) for s in CORPUS]
    vocab = vocabulary(data)
    lm1 = FirstOrderLM(data)
    lm2 = SecondOrderLM(data)

    banner("Part III: tokenised training data")
    for tokens in data:
        print(" ".join([START] + tokens + [END]))
    print(f"vocabulary ({len(vocab)} words): {vocab}")

    banner("Part IV: first-order CPT, P(next | current)")
    all_next = vocab + [END]
    for w in ["<START>", "the", "cat", "dog", "sat", "ran", "on", "to", "mat", "rug", "park"]:
        dist = lm1.probs.get(w, {})
        counts = dict(lm1.counts.get(w, {}))
        zeros = [v for v in all_next if v not in dist]
        print(f"P(. | {w}) = {{{fmt_dist(dist)}}}   counts={counts}")
        print(f"    zero-probability next words: {zeros}")
    print(f"\n'{END}' never appears as a context: P(. | {END}) is undefined (unseen context).")

    banner("Part VII: normalisation test, sum_v P(v | w) for every context w")
    for ctx, dist in lm1.probs.items():
        print(f"  {ctx:8s} {sum(dist.values()):.12f}")
    print(f"  max |total - 1| first-order  = {normalisation_test(lm1):.2e}")
    print(f"  max |total - 1| second-order = {normalisation_test(lm2):.2e}")

    # Extra check: the sampler actually follows the CPT.
    n = 200_000
    counts = Counter(lm1.sample(["the"], rng) for _ in range(n))
    print(f"\n  sampler check: {n} draws of X_t+1 given X_t = 'the'")
    for w, p in sorted(lm1.probs["the"].items()):
        print(f"    {w:5s} CPT {p:.4f}   empirical {counts[w] / n:.4f}")

    # Extra check: counts rebuild the corpus statistics exactly.
    bigrams = Counter()
    for t in data:
        seq = [START] + t + [END]
        bigrams.update(zip(seq, seq[1:]))
    assert all(lm1.counts[a][b] == c for (a, b), c in bigrams.items())
    print("  count check: every bigram count matches an independent recount")

    banner("Part VIII: next-word prediction (argmax)")
    for w in ["<START>", "the", "cat", "dog", "sat", "ran", "on", "to"]:
        dist = lm1.distribution([w])
        best = lm1.predict([w])
        top = max(dist.values())
        tied = sorted(v for v, p in dist.items() if p == top)
        note = f"  (tie between {tied}, broken alphabetically)" if len(tied) > 1 else ""
        print(f"  argmax_w P(w | {w}) = {best}  (p = {top:.4f}){note}")
    print(f"  unseen context: P(. | elephant) = {lm1.distribution(['elephant'])}, predict -> {lm1.predict(['elephant'])}")

    banner("Part IX: 25 sampled sentences, first-order model")
    first_order_samples = []
    for i in range(25):
        toks, why = lm1.generate(rng)
        first_order_samples.append(" ".join(toks))
        print(f"  {i + 1:2d}. {' '.join(toks)}   [{why}]")
    with open("generated_first_order.txt", "w") as f:
        f.write("\n".join(first_order_samples) + "\n")
    print("  saved to generated_first_order.txt")

    banner("Part X: greedy vs sampling (first-order)")
    print("  Mode A (greedy):")
    for i in range(5):
        toks, why = lm1.generate(greedy=True)
        print(f"    {i + 1}. {' '.join(toks)}   [{why}]")
    print("  Mode B (sampling):")
    for i in range(5):
        toks, why = lm1.generate(rng)
        print(f"    {i + 1}. {' '.join(toks)}   [{why}]")

    banner("Part XI/XII: second-order CPT, P(next | prev2, prev1)")
    for ctx in sorted(lm2.probs):
        print(f"  P(. | {ctx[0]}, {ctx[1]}) = {{{fmt_dist(lm2.probs[ctx])}}}")
    print(f"  unseen context: P(. | the, sat) = {lm2.distribution(['the', 'sat'])}")

    banner("Part XI: 25 sampled sentences, second-order model")
    second_order_samples = []
    for i in range(25):
        toks, why = lm2.generate(rng)
        second_order_samples.append(" ".join(toks))
        print(f"  {i + 1:2d}. {' '.join(toks)}   [{why}]")
    with open("generated_second_order.txt", "w") as f:
        f.write("\n".join(second_order_samples) + "\n")
    print("  saved to generated_second_order.txt")
    print("  second-order greedy:")
    for i in range(2):
        toks, why = lm2.generate(greedy=True)
        print(f"    {' '.join(toks)}   [{why}]")

    banner("Part XIII: comparing the two models")
    V = len(vocab)
    ctx1 = [START] + vocab                                   # possible contexts
    ctx2 = [(START, START)] + [(START, w) for w in vocab] + [(a, b) for a in vocab for b in vocab]
    out_size = V + 1                                         # vocab + <END>
    train_set = set(CORPUS)
    rows = []
    for name, model, contexts, samples in [
        ("first-order", lm1, ctx1, first_order_samples),
        ("second-order", lm2, ctx2, second_order_samples),
    ]:
        observed = sum(len(d) for d in model.probs.values())
        free = sum(len(d) - 1 for d in model.probs.values())
        unseen = sum(1 for c in contexts if c not in model.probs)
        uniq = len(set(samples))
        novel = len(set(samples) - train_set)
        rows.append((name, len(contexts) * out_size, observed, free, len(contexts), unseen, uniq, novel))
    print("| Model | full CPT size | non-zero entries | free parameters (non-zero, minus 1 per context) | possible contexts | unseen contexts | distinct sentences in 25 samples | novel (not in training data) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")

    print("\n  probability of training and novel sentences:")
    for s in ["the cat sat on the mat", "the dog ran to the park", "the cat sat on the park",
              "the dog ran to the mat", "the cat sat on the cat sat on the mat"]:
        t = tokenise(s)
        print(f"    {s:42s} first-order {lm1.sentence_probability(t):.6f}   second-order {lm2.sentence_probability(t):.6f}")

    # Chain rule check: joint = product of conditionals, and the probabilities of
    # all sentences the second-order model can produce sum to 1.
    def all_sentences(model, history, p, acc):
        for w, q in model.distribution(history).items():
            if w == END:
                acc.append((history[model.order:], p * q))
            else:
                all_sentences(model, history + [w], p * q, acc)
        return acc
    support = all_sentences(lm2, lm2.start_history(), 1.0, [])
    print(f"\n  second-order model: {len(support)} possible sentences, total probability {sum(p for _, p in support):.12f}")
    for toks, p in sorted(support, key=lambda x: -x[1]):
        print(f"    {p:.4f}  {' '.join(toks)}")


if __name__ == "__main__":
    main()
