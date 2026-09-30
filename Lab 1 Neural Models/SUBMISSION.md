# Lab 1: Neural Models (Learning, Depth, Activations, Output Layers)

Course: Artificial Intelligence (CS F407)

All numbers below come from `results.txt`, produced by running `python3 xor_lab.py` (PyTorch 2.11, CPU, fixed seeds, so the run is reproducible).

---

## Task 1: Problem specification

**Input space:** X = {0, 1} x {0, 1} (two binary sensors x1, x2).
**Output space:** Y = {0, 1} (1 = raise the disagreement warning).

| x1 | x2 | y |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Sketch of the four points**

```
x2
 1 |  (0,1) y=1        (1,1) y=0
   |
 0 |  (0,0) y=0        (1,0) y=1
   +----------------------------- x1
      0                 1
```

The class 1 points sit on one diagonal and the class 0 points sit on the other.

**Why one straight boundary cannot work:** a line w1 x1 + w2 x2 + b = 0 would need (0,1) and (1,0) on the positive side, so w2 + b > 0 and w1 + b > 0. Adding these gives w1 + w2 + 2b > 0. It would also need (0,0) and (1,1) on the negative side, so b < 0 and w1 + w2 + b < 0. Adding those gives w1 + w2 + 2b < 0. That is a contradiction, so no line separates the classes.

**Prediction for a single affine layer + sigmoid:** the best it can do is give up and output 0.5 everywhere, with loss ln 2 = 0.6931. It should get at most 2 or 3 of the 4 points "right", and only by chance.

**Result (checks the prediction):** final loss 0.693147 (exactly ln 2), all four probabilities 0.5, weights driven to about 1e-8. I also stacked three affine layers (2-8-8-1) with no activations. The loss stayed at 0.6936, and multiplying the three weight matrices out gave a single affine map that matched the network to 1.5e-8. Depth without nonlinearity adds nothing.

**Think about it:** XOR tests the claim that the *kind* of representation matters, not the number of parameters. The deep linear model had 105 parameters and still failed, because every model it can express is linear in x. The claim we can test with four points is that a nonlinear hidden layer can re-represent the inputs so that the classes become linearly separable.

---

## Task 2: Model design and validation criteria

**Design:** 2 inputs -> 2 hidden units (tanh, later sigmoid and ReLU) -> 1 output logit. Sigmoid on the output, binary cross-entropy loss (implemented as `BCEWithLogitsLoss`), full-batch Adam, learning rate 0.05, 5000 steps, fixed seed.

1. **Why the hidden nonlinearity is scientifically necessary:** without it the whole network is one affine map (shown above), and no affine map separates XOR. With a nonlinearity the hidden layer can bend the input space. In the successful run, h(1) mapped (0,1) to [0.99, -1.00] and (1,0) to [-1.00, 0.99], while (0,0) and (1,1) both went near [-0.99, -0.98]. In that hidden space a single line does separate the classes.
2. **Why sigmoid + binary cross-entropy is a good pairing:** the target is one yes/no answer, so we need one probability in (0, 1), and sigmoid gives that. BCE is the negative log-likelihood of a Bernoulli variable, so minimising it is maximum likelihood. Combined, the gradient with respect to the logit is simply p - y. That gradient does not vanish when the sigmoid saturates on a wrong answer, unlike squared error. `BCEWithLogitsLoss` fuses both steps in a numerically stable way.
3. **Evidence of successful learning (checks used):**
   - final loss well below ln 2 = 0.693 (the "predict 0.5 everywhere" level);
   - all four thresholded labels match the XOR table;
   - first-layer gradients are non-zero at the start and match a finite-difference estimate;
   - the hidden representation actually separates the classes;
   - repeated runs over many seeds, to see whether success depends on the initialisation.

**Think about it:** nothing tells the hidden units what to compute. Their "targets" come from backpropagation. The output error p - y is sent backwards through W(2) and the activation derivative, so each hidden unit gets the direction that most reduces the final loss. The hidden features are whatever makes the output layer's job easiest. The loss on the output, and nothing else, decides them.

---

## Task 3: Using an LLM to generate the implementation

**Prompt used (Claude):**

> Generate minimal PyTorch code for the following model and dataset. Do not change the architecture or task.
> Dataset: the four XOR examples (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0.
> Model: a 2-2-1 network with a tanh hidden activation and a single output logit trained with BCEWithLogitsLoss. Use PyTorch's default random initialisation.
> Train full-batch with Adam for 5000 steps on CPU.
> After training, report the final loss, all four probabilities, thresholded labels, and one parameter-gradient tensor (the first-layer weight gradient after backward()). Set a random seed for reproducibility and explain each test in one sentence.

Follow-up prompts asked for the zero-initialisation copy (Part C), a loop over sigmoid/tanh/ReLU (Part D), and "modify only the output layer and loss" for the three-class version (Task 5).

**Inspection before running** (in `xor_lab.py`, function `train`):
- forward pass: `logits = model(X)`, which calls `XORNet.forward` (`hidden`, then `act`, then `out`);
- scalar loss: `loss = loss_fn(logits, targets)`;
- reverse-mode AD: `loss.backward()`;
- parameter update: `opt.step()`, after `opt.zero_grad()` clears the previous gradients.

**Changes made to the generated code:**
1. The minimal version printed one gradient tensor with no evidence it was correct. Before running, I added a central finite-difference check in float64 and a per-example gradient average (Task 4B).
2. One seed is weak evidence for a 2-2-1 XOR network, so I added a 20-seed repeated-run table (Task 4D).
3. After the first run, the all-zero experiment showed gradients that were exactly zero. That is a saddle point, not the symmetry effect the lab asks about. I added a second symmetric case with every weight set to 0.5 (Task 4C).

The output stays a raw logit with `BCEWithLogitsLoss` (not `Sigmoid` + `BCELoss`) for numerical stability. Sigmoid is applied only when reporting probabilities.

**Think about it:** some things can be checked by reading the code alone: the architecture (layer shapes), the loss function, whether the sigmoid is applied twice, whether `zero_grad` is called, and whether the task labels are right. Other things need execution and measurement: whether training converges, whether the gradients are numerically correct, whether all four labels come out right, and how sensitive the result is to the seed.

---

## Task 4: Execution, testing, diagnosis

### Part A: Basic learning check (tanh, seed 0)

| | value |
|---|---|
| initial loss | 0.715159 |
| final loss | 0.000024 |

| x | P(y=1) | predicted | target |
|---|---|---|---|
| (0,0) | 0.0000 | 0 | 0 |
| (0,1) | 1.0000 | 1 | 1 |
| (1,0) | 1.0000 | 1 | 1 |
| (1,1) | 0.0000 | 0 | 0 |

All four labels are correct. No settings had to be changed for this seed.

### Part B: Backpropagation check

At initialisation (seed 0), `model.hidden.weight.grad` is

```
[[ 0.0005,  0.0006],
 [-0.0426, -0.0448]]
```

Entry (i, j) is dL/dW(1)[i, j]: how much the mean loss changes per unit change of the weight from input j to hidden unit i, with everything else held fixed. A central finite difference, (L(w + eps) - L(w - eps)) / 2eps with eps = 1e-6, agreed with autograd to within **5.4e-11**.

The loss is the **mean** over the 4 examples, L = (1/4) sum_k L_k. Differentiation is linear, so dL/dW = (1/4) sum_k dL_k/dW: the gradient is the average of the per-example gradients. I checked this directly by back-propagating each example on its own and averaging. The difference was **0.0** (exact).

### Part C: Symmetry experiment

**All weights and biases = 0 (tanh and sigmoid):** the two rows of W(1) stayed identical ([0, 0] and [0, 0]) at every step checked (0, 1, 2, 5, 10, 100, 3000). The loss never moved from 0.6931, and every output was 0.5. The gradient for W(1) was exactly zero. W(2) = 0 means dL/dh = W(2)^T (p - y) = 0, so nothing flows back to the first layer. The network sits exactly on a saddle point.

**All weights equal to 0.5 (non-zero but symmetric):** this makes the symmetry argument visible with non-zero gradients:

| step | W(1) row 0 | W(1) row 1 | grad row 0 = grad row 1 | loss |
|---|---|---|---|---|
| 0 | [0.5, 0.5] | [0.5, 0.5] | [-0.00221, -0.00221] | 0.7038 |
| 10 | [0.9877, 0.9877] | [0.9877, 0.9877] | [-0.01509, -0.01509] | 0.6540 |
| 3000 | [7.1221, 7.1221] | [7.1221, 7.1221] | [-1e-05, -1e-05] | 0.4774 |

The rows change but stay identical, and training ends with 3/4 correct (loss 0.4774).

**Explanation:** if the two hidden units start with the same incoming weights and the same outgoing weight, they compute the same output for every input. They therefore receive the same error signal, and so the same gradient. After the update they are still identical, and by induction they stay identical forever. The network behaves as if it had one hidden unit, and one hidden unit cannot solve XOR. That is exactly the 3/4 result. Random initialisation breaks this symmetry.

### Part D: Activation experiment (same seed and same initial weights for all three)

| Hidden activation | Final loss | 4/4 correct? | Early ‖∇W(1) L‖₂ (step 0) | Early ‖∇W(1) L‖₂ (step 10) |
|---|---|---|---|---|
| Sigmoid | 0.477397 | no | 0.0009 | 0.0012 |
| Tanh | 0.000024 | yes | 0.0618 | 0.0094 |
| ReLU | 0.693148 | no | 0.0017 | 0.0000 |

Repeated runs (seeds 0 to 19, same settings):

| Hidden activation | runs with 4/4 correct | mean final loss |
|---|---|---|
| Sigmoid | 8/20 | 0.2342 |
| Tanh | 9/20 | 0.2037 |
| ReLU | 6/20 | 0.4097 |

**Interpretation:** for this seed, tanh had by far the largest early first-layer gradient (0.0618, against 0.0009 for sigmoid). Its derivative at 0 is 1, while sigmoid's is at most 0.25, and tanh is zero-centred. Tanh converged to a near-zero loss. Sigmoid learned slowly at first and settled into a local minimum with loss 0.477. Its final pre-activations were large (for example -25.8 and 28.8), so its units saturated with derivatives near zero. ReLU failed for a different reason: by step 10 the gradient was 0.0000, and at the end every hidden pre-activation was negative for all four inputs (between -0.2 and -1.5). Both units were dead, so the network output a constant and the loss stayed at ln 2. The multi-seed table shows these are properties of this tiny 2-2-1 problem and its initialisation, not a ranking of activations. With only two hidden units, XOR has local minima, and every activation fails on more than half the seeds. Four data points do not show that one activation is universally best.

**Think about it:** to tell the two mechanisms apart, look at the **pre-activations** a(1), not just the gradient. A saturated sigmoid has a large |a| (here |a| > 12), and its activation h sits at nearly 0 or 1. A dead ReLU has a < 0 for every input, and h is exactly 0. The derivative is then exactly 0, not just small, and it stays 0 because no input ever makes a positive. The ReLU table above shows exactly this pattern.

---

## Task 5: Three-class extension

Only the output layer and loss changed: `nn.Linear(2, 3)` and `nn.CrossEntropyLoss` (log-softmax + negative log-likelihood).

**Predictions before running:**
1. Final weight matrix W(2) shape: **(3, 2)** (3 classes, 2 hidden units). Confirmed: `(3, 2)`.
2. Logits per example: **3**. Confirmed.
3. Softmax probabilities sum to one because p_k = exp(z_k) / sum_j exp(z_j). Every term is positive and they share the same denominator, so the sum is sum_k exp(z_k) / sum_j exp(z_j) = 1.
4. Why the gradient is p - y: L = -log p_c = -z_c + log sum_j exp(z_j). Differentiating with respect to z_k gives -[k = c] + exp(z_k) / sum_j exp(z_j) = p_k - y_k.

**Results:** initial loss 1.059 (close to ln 3 = 1.0986, as expected for a near-uniform start). Final loss 0.000011.

| x | P(class 0) | P(class 1) | P(class 2) | predicted | target |
|---|---|---|---|---|---|
| (0,0) | 1.0000 | 0.0000 | 0.0000 | 0 | 0 |
| (0,1) | 0.0000 | 1.0000 | 0.0000 | 1 | 1 |
| (1,0) | 0.0000 | 1.0000 | 0.0000 | 1 | 1 |
| (1,1) | 0.0000 | 0.0000 | 1.0000 | 2 | 2 |

**Normalisation check** for x = (0,1): p = [6.634e-06, 0.9999914, 1.942e-06], and the sum is 1.0000000000.

**p - y check:** the autograd gradient of the cross-entropy with respect to the logits matched softmax(z) - onehot(y) to within 2.3e-13.

**Shift invariance (optional diagnostic):** adding 100 to all three logits changed the probabilities by at most 3.6e-12 (float32 round-off), because exp(z_k + c) / sum exp(z_j + c) = exp(z_k) / sum exp(z_j). With a shift of 1000, the naive exp overflows to [inf, inf, inf] even in float64, and inf/inf is NaN. Subtracting the max logit first gives the correct vector. Stable implementations subtract the max because the shift does not change the answer, and afterwards the largest exponent is exp(0) = 1, so nothing can overflow.

**Think about it:** at a vocabulary of tens of thousands, the maths is the same: the logits -> softmax -> cross-entropy pipeline, the normalisation property, the shift invariance and max-subtraction trick, and the p - y gradient all carry over. What changes dramatically is the surrounding architecture. The input is a variable-length sequence of tokens, not two bits. Tokens need learned embeddings, and a Transformer (attention, many layers, residual connections) builds the context representation. The output matrix becomes (vocab x d_model), often tied to the embeddings. The softmax over the full vocabulary becomes a real compute and memory cost.

---

## Reflection questions

1. **Depth vs nonlinearity:** depth alone did nothing. A 3-layer linear network collapsed to one affine map and stayed at loss ln 2. Two hidden tanh units (less depth, fewer parameters) solved XOR. The capacity to represent XOR comes from the nonlinearity between layers; depth only helps when there is a nonlinearity to compose.
2. **Evidence that backprop gave a useful signal, not just a non-zero one:** (a) the gradient matched finite differences to 5e-11, so it was the true gradient of the loss; (b) following it drove the loss from 0.715 to 0.000024; (c) it produced an interpretable hidden representation in which each disagreement input switched on a different hidden unit; (d) all four labels came out correct. In contrast, the saturated-sigmoid run had non-zero gradients but reached only a local minimum (loss 0.477).
3. **Why identical/zero init prevents distinct features:** identical units compute identical outputs, receive identical gradients, and receive identical updates, so they can never diverge (the "weight 0.5" run kept both rows equal for 3000 steps). All-zero init is worse still: W(2) = 0 blocks the backward signal completely, and the gradient is exactly zero.
4. **Effect of activation on the gradient:** *Engineering observation:* at the same initialisation, the early ‖∇W(1)‖ was 0.0618 for tanh, 0.0009 for sigmoid, and 0.0017 for ReLU, falling to 0.0000 by step 10. *Scientific explanation:* the backward pass multiplies by f'(a). Tanh's maximum slope is 1 and it is centred at 0. Sigmoid's slope is at most 0.25, and near 0 when saturated. ReLU's slope is exactly 0 for negative inputs, so units that go negative for every input stop learning altogether.
5. **Why the output layer and loss go together:** the loss must be the negative log-likelihood of the distribution that the output layer parameterises. Sigmoid parameterises a Bernoulli, so the loss is BCE. Softmax parameterises a categorical, so the loss is cross-entropy. Choosing them together makes the logit gradient p - y, which is well-scaled and does not vanish when confidently wrong. A mismatched pair (such as softmax with squared error, or BCE on softmax outputs) is either the wrong likelihood or gives saturating gradients.
6. **LLM productivity vs human verification:** *Productivity:* the LLM produced the training loop, the model class, and the `BCEWithLogitsLoss` / `CrossEntropyLoss` wiring in seconds, including the stable logit-based loss. *Human verification essential:* the first draft's single-seed run made one activation look "best". Only the 20-seed table showed that every activation fails on more than half the seeds. Likewise, the all-zero experiment looked like "symmetry", but gradients that are exactly zero are a different (saddle) phenomenon. I had to add the non-zero symmetric case to show what the lab asks about.
7. **Which tests scale:** keep loss curves, prediction checks on held-out data, gradient norm monitoring per layer, activation statistics (the fraction of dead ReLUs, saturation), multi-seed runs where affordable, and softmax normalisation / shift checks. Exhaustive finite-difference gradient checks become too expensive: they need 2 forward passes per parameter, which is billions of passes for a large model. At scale you only spot-check a few random parameters on a tiny model or batch. Exhaustive per-example gradient comparisons also become too expensive.
