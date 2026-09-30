# Lab 7: Bayesian Networks and Autoregressive Language Models

Course: Artificial Intelligence (CS F407)

Everything below comes from `results.txt`, produced by `python3 ngram_lm.py` (plain Python, fixed random seed 407). Generated text is saved in `generated_first_order.txt` and `generated_second_order.txt`.

---

## Part I

**Question 1: Why is the chain-rule decomposition useful for generating text?**
It turns one impossible-to-estimate joint distribution over whole sentences into a sequence of *next-token* distributions, each conditioned only on what has already been produced. Generation then becomes a simple loop: sample X1, then sample X2 given X1, then X3 given X1 and X2, and so on until END. At every step we only need one conditional distribution over the vocabulary, not over all possible sentences. The decomposition is exact (no assumption yet), and the same factors also give the probability of any complete sentence, as the product of the conditionals.

---

## Part II

**Question 2: Independence assumption of X1 -> X2 -> X3 -> X4.**
Each word is conditionally independent of all earlier words, given the immediately preceding word (the first-order Markov assumption):

P(X_t | X_1, ..., X_{t-1}) = P(X_t | X_{t-1}), which is the same as saying X_t ⊥ {X_1, ..., X_{t-2}} | X_{t-1}.

So P(X1, X2, X3, X4) = P(X1) P(X2 | X1) P(X3 | X2) P(X4 | X3).

---

## Part III: Dataset

The six handout sentences, lower-cased, whitespace-tokenised, and wrapped in `<START>` ... `<END>`. Vocabulary (10 words): cat, dog, mat, on, park, ran, rug, sat, the, to.

---

## Part IV: Conditional probability tables

**Question 3: P(next word | current word)**, estimated as C(w_i, w_j) / sum_k C(w_i, w_k):

| current | next-word distribution | counts |
|---|---|---|
| `<START>` | the 1.0 | the 6 |
| the | cat 0.25, dog 0.25, mat 0.1667, park 0.1667, rug 0.1667 | cat 3, dog 3, mat 2, park 2, rug 2 |
| cat | sat 0.6667, ran 0.3333 | sat 2, ran 1 |
| dog | sat 0.6667, ran 0.3333 | sat 2, ran 1 |
| sat | on 1.0 | on 4 |
| ran | to 1.0 | to 2 |
| on | the 1.0 | the 4 |
| to | the 1.0 | the 2 |
| mat / rug / park | `<END>` 1.0 | 2 each |

"the" occurs 12 times as a context: 6 times sentence-initially (followed by cat or dog) and 6 times after on or to (followed by mat, rug or park). So P(cat | the) = 3/12 = 0.25. The handout's 3/5 was a hypothetical example.

**Zero-probability transitions** (all of the following have count 0, so probability 0):
- after **the**: on, ran, sat, the, to, `<END>`;
- after **cat** and **dog**: everything except sat and ran (for example P(on | cat) = 0 and P(`<END>` | cat) = 0);
- after **sat**: everything except on (for example P(to | sat) = 0);
- after **ran**: everything except to (for example P(on | ran) = 0);
- after on, to, and `<START>`: everything except the;
- after mat, rug, park: every word (only `<END>` follows).

Of the 11 x 11 = 121 possible (context, next) pairs, only **17** are non-zero.

---

## Part V: Prompt given to the LLM (Claude)

> Write a simple Python implementation of a first-order autoregressive language model. The model should:
> 1. take a list of tokenised sentences as training data;
> 2. count transitions between consecutive tokens;
> 3. construct the conditional distribution P(X_t | X_{t-1});
> 4. display the probabilities for a specified previous token;
> 5. predict the most probable next token;
> 6. generate a sentence by repeatedly sampling the next token;
> 7. stop when the `<END>` token is generated.
>
> Do not use a machine-learning library or a pretrained language model. Use ordinary Python data structures and random sampling. Add `<START>` and `<END>` markers to each sentence. Support two generation modes, greedy (argmax) and sampling. Use a seeded random.Random so runs are reproducible.

---

## Part VI: Inspecting the code (`ngram_lm.py`)

**Question 4: Where are the transition counts stored?**
In `FirstOrderLM.__init__`, in `self.counts`, a `defaultdict(Counter)`. `self.counts[prev][next]` is C(prev, next), filled by walking over `zip(seq, seq[1:])` for each padded sentence.

**Question 5: Where is P(X_t | X_{t-1}) computed?**
Also in `__init__`, right after counting. For each context, `self.probs[prev] = {w: c / total ...}`, where `total = sum(nexts.values())`. `distribution(history)` looks this up for the last token.

**Question 6: How is the next word chosen?**
Both ways are supported. `predict()` always picks the argmax (greedy, with ties broken alphabetically). `sample()` draws from the distribution with `rng.choices(words, weights=probs)`. The greedy choice is deterministic: from the same context it always returns the same word, so it can only ever produce one sentence. Sampling picks each word with a frequency proportional to its probability, so lower-probability continuations also appear and repeated runs differ. I checked the sampler empirically: over 200,000 draws after "the", the frequencies were cat 0.2492, dog 0.2502, mat 0.1666, park 0.1665, rug 0.1676, against CPT values of 0.25, 0.25 and 0.1667.

**Question 7: What happens with an unobserved context?**
`self.probs.get(context, {})` returns an empty distribution. `predict()` and `sample()` then return `None`, and `generate()` stops with the reason `"unseen context"`. For example, `P(. | elephant) = {}` and `predict -> None`. Without this guard, `rng.choices([], weights=[])` would raise an `IndexError`. `<END>` itself is never a context, so it is also "unseen", but generation stops before reaching it. This is the **zero-frequency problem** of maximum-likelihood n-grams: the model has no information at all about unseen contexts. Real systems use smoothing (such as add-one or Kneser-Ney) or back off to a shorter context.

---

## Part VII: Normalisation test

For every context w, sum over v of P(v | w):

```
<START> 1.000000000000    the 1.000000000000    cat 1.000000000000
sat     1.000000000000    on  1.000000000000    mat 1.000000000000
rug     1.000000000000    dog 1.000000000000    ran 1.000000000000
to      1.000000000000    park 1.000000000000
max |total - 1| first-order  = 0.00e+00
max |total - 1| second-order = 0.00e+00
```

Two further checks pass: every bigram count matches an independent recount from the raw sentences, and the probabilities of all sentences the second-order model can generate sum to exactly 1.000000000000 (6 sentences, each 1/6).

**Question 8: What would a total of 0.87 mean?**
The distribution for that context is not normalised, so the implementation is wrong. 13% of the probability mass is missing. Likely causes:
- the denominator counts something that is not in the numerator (for example, counting a word's total occurrences, including sentence-final ones, while not counting its transitions to `<END>`);
- some transitions are dropped (for example, the loop stops one token early and misses the `<END>` transition);
- rounding or truncation of the probabilities;
- a smoothing scheme that adds mass to the denominator without adding it to the numerators.

Sampling from such a table would be quietly biased, and sentence probabilities would be systematically wrong.

---

## Part VIII: Next-word prediction

| context | P(X_{t+1} given the context) | argmax |
|---|---|---|
| `<START>` | the 1.0 | the |
| the | cat .25, dog .25, mat .167, park .167, rug .167 | **cat/dog tie** (cat chosen alphabetically) |
| cat | sat .667, ran .333 | sat |
| dog | sat .667, ran .333 | sat |
| sat | on 1.0 | on |
| ran | to 1.0 | to |
| on | the 1.0 | the |
| to | the 1.0 | the |

**Question 9: Are the predictions what I'd expect?**
Mostly, but not always. After "the" the model rates cat, dog, mat, park and rug as roughly equally likely, whatever came before. I would expect "the" at the start of a sentence to be followed by an animal, and "the" after "on" to be followed by a place. The model can't tell these two uses of "the" apart, because it only sees one word of context. It also has an exact tie between cat and dog, which a human would not consider a meaningful fact. A probability model encodes *co-occurrence frequencies in its training data under its independence assumptions*, while human expectations come from meaning, grammar, and far more context. The model is "right" relative to its data and assumptions, not relative to language.

---

## Part IX: Generated text (first-order, 25 samples)

From `generated_first_order.txt`:

```
the mat
the cat sat on the park
the cat sat on the dog sat on the cat ran to the mat
the mat
the dog ran to the dog ran to the cat sat on the cat sat on the dog ran to the dog ran to the dog ran to the cat   [hit 30-token cap]
the cat ran to the cat ran to the cat sat on the mat
the park
the dog ran to the dog sat on the park
the dog sat on the mat
...
the cat ran to the dog sat on the mat
the rug
```

Observations: 12 distinct sentences out of 25, and 10 of those are not in the training data. Many are ungrammatical: "the mat" as a whole sentence, "sat on the park", and loops like "sat on the dog sat on the cat". "the mat" / "the park" / "the rug" come up often: after `<START>` "the" is certain, and then 3 of the 5 continuations of "the" lead straight to `<END>` (probability 0.5 in total).

---

## Part X: Greedy vs sampling

**Mode A (greedy), 5 runs:** all five were identical:

```
the cat sat on the cat sat on the cat sat on ... the cat     [hit the 30-token cap]
```

**Mode B (sampling), 5 runs:**

```
the dog sat on the dog sat on the cat sat on the rug
the rug
the park
the cat sat on the cat sat on the rug
the dog ran to the mat
```

**Question 10:** sampling gives far more variation. Greedy decoding is a deterministic function of the context, so from `<START>` it always follows the same chain. In this model that chain is also a cycle: the -> cat (tie) -> sat -> on -> the -> cat ..., because "mat", "rug" and "park" (the only ways out to `<END>`) each have lower probability after "the" than cat or dog. Greedy never terminates on its own, which is why the generator needs a length cap. Sampling explores the whole distribution: it produced 5 different sentences in 5 runs, and it does reach `<END>`, because at each "the" there is a 0.5 chance of picking a word that ends the sentence.

---

## Part XI and XII: Second-order model

**Question 11: How the second-order model differs**

1. **Graph structure:** each X_t now has two parents, X_{t-2} -> X_t <- X_{t-1}, instead of one (X_{t-1} -> X_t). There are more edges, and the model makes a weaker independence assumption: X_t ⊥ X_{1..t-3} | X_{t-2}, X_{t-1}.
2. **CPT:** it is indexed by a *pair* of previous tokens, so it has |V|^2 possible rows instead of |V| (111 possible contexts vs 11 here, and 1221 vs 121 cells).
3. **Context:** two words instead of one. For example, it can tell apart "`<START>` the" (next word: cat or dog) from "on the" (next word: mat or rug) and "to the" (next word: park).
4. **Data needed:** far more. Each of the many more rows must be estimated from the (fewer) occurrences of that exact pair. 96 of the 111 possible contexts were never seen.

**Prompt for Part XII (Claude):**

> Modify the existing first-order autoregressive model into a second-order model. The model should estimate P(X_t | X_{t-2}, X_{t-1}). Represent the model using counts of observed triples and use these counts to construct conditional probability distributions. Pad each sentence with two `<START>` tokens so the first two words have a defined context. Keep the same interface (distribution, predict, sample, generate). Do not replace the model with a neural network or a pretrained language model.

**What should change, stated before accepting the code:** only the context. The counts become `counts[(w_{t-2}, w_{t-1})][w_t]`, the padding becomes two `<START>`s, and `context(history)` returns the last two tokens. Normalisation, sampling, greedy prediction and the stopping rule should stay identical. The generated `SecondOrderLM` overrides exactly `__init__` and `context` and inherits everything else, which matches that expectation.

**Second-order CPT (all observed contexts):**

| context | distribution |
|---|---|
| (`<START>`, `<START>`) | the 1.0 |
| (`<START>`, the) | cat 0.5, dog 0.5 |
| (the, cat), (the, dog) | sat 0.667, ran 0.333 |
| (cat, sat), (dog, sat) | on 1.0 |
| (cat, ran), (dog, ran) | to 1.0 |
| (sat, on) | the 1.0 |
| (ran, to) | the 1.0 |
| (on, the) | mat 0.5, rug 0.5 |
| (to, the) | park 1.0 |
| (the, mat), (the, rug), (the, park) | `<END>` 1.0 |

The first-order model's single "the" row is split into three rows with different meanings. 25 samples (`generated_second_order.txt`):

```
the dog ran to the park
the cat ran to the park
the cat sat on the rug
the dog sat on the mat
...
the cat sat on the mat
```

Greedy second-order generation gives `the cat sat on the mat` and terminates.

---

## Part XIII: Comparing the models

| | first-order | second-order |
|---|---|---|
| full CPT size (contexts x outcomes) | 11 x 11 = 121 | 111 x 11 = 1221 |
| non-zero entries | 17 | 19 |
| free parameters (non-zero minus one per context) | 6 | 4 |
| possible contexts | 11 | 111 |
| **unseen (zero-information) contexts** | **0** | **96** |
| distinct sentences in 25 samples | 12 | 6 |
| novel sentences (not in training data) | 10 | 0 |
| sentences with non-zero probability | infinitely many (the graph has cycles) | exactly 6 (the training sentences, 1/6 each) |

**Probabilities of specific sentences:**

| sentence | first-order | second-order |
|---|---|---|
| the cat sat on the mat | 0.0278 | 0.1667 |
| the dog ran to the park | 0.0139 | 0.1667 |
| the cat sat on the park | 0.0278 | 0 |
| the dog ran to the mat | 0.0139 | 0 |
| the cat sat on the cat sat on the mat | 0.0046 | 0 |

**Qualitative coherence:** every second-order sentence is grammatical and sensible, but that is because it can only *reproduce the training sentences*. The first-order model is creative but incoherent: "the cat sat on the park", "the mat", and loops. It gives the ungrammatical "the cat sat on the park" the same probability as the real "the cat sat on the mat".

**Question 12:** more context lets the model condition on information that actually determines the next word. "on the" is followed by mat or rug, while "`<START>` the" is followed by cat or dog. This sharpens the distributions and removes absurd continuations, and the sentence probabilities show it: 0.167 vs 0.028 for a real sentence, 0 vs 0.028 for a bad one. But the CPT grows as |V|^k for k words of context (121 cells, then 1221, and a third-order table would need on the order of 11^3 rows of 11 entries each), while the amount of data stays fixed. Most rows are then estimated from very few examples or none at all (96 of 111 contexts unseen), so maximum-likelihood estimates overfit. Here the second-order model memorised the corpus exactly and gives probability 0 to any new sentence. More context lowers bias but raises variance and data sparsity. That trade-off is the curse of dimensionality in the CPT size.

---

## Part XIV: Connection to modern LMs

A neural LM keeps exactly the same objective and factorisation. It replaces the lookup table P(X_t | x_{t-k..t-1}) with a function f_theta(x_1..x_{t-1}) that outputs a softmax over the vocabulary. The function generalises across contexts through shared parameters and embeddings, so it does not need a separate row for every context. Generation still means sampling from that conditional distribution.

---

## Part XV

**Question 13: Why is Approach B better than "write me a language model"?**
- *Specifying the intended behaviour:* B names the exact object to build, P(X_t | X_{t-1}) from counts with sampling-based generation. That gives a definition of "correct" before any code exists. Approach A lets the LLM choose the model: maybe a bigram, maybe a call to a pretrained transformer. We would not know what we got.
- *Understanding the representation:* with B, I know the model *is* a table of counts and normalised rows. I could predict P(cat | the) = 0.25 by hand before running anything, and use it to check the code.
- *Validating the implementation:* a specification gives testable consequences. The counts must match a recount, the rows must sum to 1, samples must match CPT frequencies, and sentence probabilities must equal the product of the conditionals. With Approach A there is nothing to check against.
- *Testing probabilistic invariants:* the normalisation test, the sampler frequency test, and "all sentence probabilities sum to 1" are properties of the *model*, not of the code. They catch bugs (for example a missing `<END>` transition gives row sums below 1) that simply running the program would not reveal.
- *Distinguishing implementation from model:* B keeps the scientific object (the BN and its CPTs) separate from how it is coded. The LLM is then a translator from a known model to Python, not the author of an unknown one. That separation also let me swap first-order for second-order by changing *only* the context definition.

---

## Final question

**Question 14: What did viewing the LM as a Bayesian network give me?**
- **A representation of dependencies:** the graph X_{t-1} -> X_t shows exactly what each word is allowed to depend on. Adding the edge X_{t-2} -> X_t is the whole difference between the two models.
- **A factorisation of the joint:** the joint probability of a sentence is the product of the local CPT entries. That gave me sentence probabilities (0.0278 vs 0.1667) and the test that all sentences' probabilities sum to 1.
- **A principled method for generation:** ancestral sampling in topological order (sample each node given its already-sampled parents) *is* autoregressive text generation.
- **A way to reason about independence assumptions:** the first-order model's failure to distinguish sentence-initial "the" from "on the" follows directly from the conditional independence X_t ⊥ X_{t-2} | X_{t-1}.
- **A way to understand more context:** more parents means exponentially larger CPTs, which explains the 96 unseen contexts and the memorisation.
- **A way to test the implementation:** each CPT column must be a distribution, each entry must equal a ratio of counts, and samples must match the CPT. Those tests come straight from the BN semantics.

---

## Reflection on using the LLM

The LLM translated my specification into working code quickly. That covered counting with `defaultdict(Counter)`, normalising, `random.choices` for sampling, and a clean subclass for the second-order model. I validated it by (i) computing several CPT entries by hand before running (P(cat | the) = 3/12, P(sat | cat) = 2/3); (ii) the normalisation test; (iii) an independent bigram recount; (iv) a 200,000-draw empirical check that the sampler matches the CPT; and (v) checking that the second-order model's sentence probabilities sum to 1.

**Code I inspected and corrected:** the specification's stopping rule ("stop when `<END>` is generated") is not a termination guarantee. Tracing the greedy path by hand showed the -> cat -> sat -> on -> the is a cycle that never reaches `<END>`. So greedy generation as literally specified runs forever. I added a `MAX_LEN` cap and a returned stop reason, and the experiment confirms the problem: all five greedy runs hit the cap. Inspection also showed that `rng.choices` on an unseen context would crash with an `IndexError`. I added the empty-distribution guard, and `generate` now stops with the reason "unseen context". Finally, `max(dist, key=dist.get)` breaks the cat/dog tie by dictionary insertion order, which is an accident of the data order. I made the tie-break explicit (alphabetical) so greedy output is reproducible and documented.
