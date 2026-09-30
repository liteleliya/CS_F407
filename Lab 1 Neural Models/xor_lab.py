"""
CS F407 Lab: Neural Models (Learning, Depth, Activations, Output Layers)

Runs every experiment required by the lab handout, in order:

  Task 1   linear baseline (affine + sigmoid) on XOR, to show it cannot fit
  Task 4A  2-2-1 network with tanh hidden layer: losses, probabilities, labels
  Task 4B  backpropagation check: autograd gradient vs finite differences,
           and mean loss gradient == average of per-example gradients
  Task 4C  symmetry experiment with all weights initialised to zero
  Task 4D  activation experiment (sigmoid, tanh, ReLU) plus a multi-seed
           success rate, since one seed is weak evidence
  Task 5   three-class extension with softmax / cross-entropy, including the
           p - y gradient check and the logit shift invariance check

Usage:
    python3 xor_lab.py
"""

import torch
import torch.nn as nn

torch.set_printoptions(precision=4, sci_mode=False)

# The four XOR examples. y is the disagreement warning.
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# Three-class targets for Task 5: 0 = both off, 1 = disagree, 2 = both on.
Y3 = torch.tensor([0, 1, 1, 2])

ACTIVATIONS = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


class XORNet(nn.Module):
    """2 inputs -> 2 hidden units (nonlinear) -> n_out logits."""

    def __init__(self, activation="tanh", n_hidden=2, n_out=1):
        super().__init__()
        self.hidden = nn.Linear(2, n_hidden)       # W1: (n_hidden, 2), b1: (n_hidden,)
        self.act = ACTIVATIONS[activation]()
        self.out = nn.Linear(n_hidden, n_out)      # W2: (n_out, n_hidden), b2: (n_out,)

    def forward(self, x):
        a1 = self.hidden(x)        # pre-activation a(1) = W1 x + b1
        h1 = self.act(a1)          # hidden representation h(1) = f(a(1))
        return self.out(h1)        # logits (no sigmoid here, the loss applies it)


def train(model, loss_fn, targets, steps=5000, lr=0.05, record_grad_at=10, log=False):
    """Full-batch Adam training. Returns (initial_loss, final_loss, early_grad_norm)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    initial_loss = None
    early_grad_norm = None
    for step in range(steps):
        opt.zero_grad()
        logits = model(X)                          # forward pass
        loss = loss_fn(logits, targets)            # scalar loss (mean over 4 examples)
        loss.backward()                            # reverse-mode AD fills .grad
        if step == 0:
            initial_loss = loss.item()
        if step == record_grad_at:
            early_grad_norm = model.hidden.weight.grad.norm().item()
        if log and step % 1000 == 0:
            print(f"  step {step:5d}  loss {loss.item():.6f}")
        opt.step()                                 # optimiser updates parameters
    with torch.no_grad():
        final_loss = loss_fn(model(X), targets).item()
    return initial_loss, final_loss, early_grad_norm


def binary_report(model):
    with torch.no_grad():
        probs = torch.sigmoid(model(X)).squeeze(1)
    labels = (probs >= 0.5).long()
    for x, p, lab, y in zip(X.tolist(), probs.tolist(), labels.tolist(), Y.squeeze(1).tolist()):
        print(f"  x={x}  P(y=1)={p:.4f}  predicted={lab}  target={int(y)}")
    return bool((labels == Y.squeeze(1).long()).all())


# ---------------------------------------------------------------------------
# Task 1: linear baseline
# ---------------------------------------------------------------------------
def task1_linear_baseline():
    section("Task 1: single affine layer + sigmoid (no hidden layer)")
    torch.manual_seed(0)
    linear = nn.Linear(2, 1)
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(linear.parameters(), lr=0.05)
    for _ in range(5000):
        opt.zero_grad()
        loss = bce(linear(X), Y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        probs = torch.sigmoid(linear(X)).squeeze(1)
    print(f"  final loss = {loss.item():.6f}   (ln 2 = {torch.log(torch.tensor(2.0)).item():.6f})")
    print(f"  probabilities = {probs.tolist()}")
    print(f"  learned w = {linear.weight.data.tolist()}, b = {linear.bias.data.tolist()}")
    correct = ((probs >= 0.5).long() == Y.squeeze(1).long()).sum().item()
    print(f"  correctly classified: {correct}/4")

    # Stacking affine layers without a nonlinearity is still one affine map.
    torch.manual_seed(0)
    deep_linear = nn.Sequential(nn.Linear(2, 8), nn.Linear(8, 8), nn.Linear(8, 1))
    opt = torch.optim.Adam(deep_linear.parameters(), lr=0.05)
    for _ in range(5000):
        opt.zero_grad()
        loss = bce(deep_linear(X), Y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        probs = torch.sigmoid(deep_linear(X)).squeeze(1)
        W = deep_linear[2].weight @ deep_linear[1].weight @ deep_linear[0].weight
        b = deep_linear[2].weight @ (deep_linear[1].weight @ deep_linear[0].bias + deep_linear[1].bias) + deep_linear[2].bias
        collapsed = X @ W.T + b
        max_diff = (collapsed - deep_linear(X)).abs().max().item()
    print("\n  3 stacked affine layers (2-8-8-1, no activations):")
    print(f"  final loss = {loss.item():.6f}, probabilities = {probs.tolist()}")
    print(f"  equivalent single affine map W = {W.tolist()}, b = {b.tolist()}")
    print(f"  max |collapsed map - network| = {max_diff:.2e}  (the depth adds nothing)")


# ---------------------------------------------------------------------------
# Task 4A: basic learning check
# ---------------------------------------------------------------------------
def task4a_basic(seed=0):
    section(f"Task 4A: 2-2-1 tanh network, BCEWithLogitsLoss, Adam lr=0.05, seed={seed}")
    torch.manual_seed(seed)
    model = XORNet("tanh")
    init_loss, final_loss, _ = train(model, nn.BCEWithLogitsLoss(), Y, log=True)
    print(f"  initial loss = {init_loss:.6f}")
    print(f"  final loss   = {final_loss:.6f}")
    ok = binary_report(model)
    print(f"  all four correct: {ok}")
    with torch.no_grad():
        h = torch.tanh(model.hidden(X))
    print("  learned hidden representation h(1) for each input:")
    for x, row in zip(X.tolist(), h.tolist()):
        print(f"    x={x}  h=[{row[0]: .4f}, {row[1]: .4f}]")
    return model


# ---------------------------------------------------------------------------
# Task 4B: backpropagation check
# ---------------------------------------------------------------------------
def task4b_backprop(seed=0):
    section("Task 4B: backpropagation check on first-layer weights W(1)")
    torch.manual_seed(seed)
    model = XORNet("tanh").double()
    Xd, Yd = X.double(), Y.double()
    bce = nn.BCEWithLogitsLoss()

    model.zero_grad()
    loss = bce(model(Xd), Yd)
    loss.backward()
    autograd = model.hidden.weight.grad.clone()
    print(f"  loss at initialisation = {loss.item():.6f}")
    print(f"  W(1).grad (dL/dW(1)) from autograd:\n{autograd}")

    # Central finite differences on each entry of W1.
    eps = 1e-6
    fd = torch.zeros_like(autograd)
    with torch.no_grad():
        for i in range(2):
            for j in range(2):
                orig = model.hidden.weight[i, j].item()
                model.hidden.weight[i, j] = orig + eps
                lp = bce(model(Xd), Yd).item()
                model.hidden.weight[i, j] = orig - eps
                lm = bce(model(Xd), Yd).item()
                model.hidden.weight[i, j] = orig
                fd[i, j] = (lp - lm) / (2 * eps)
    print(f"  finite-difference estimate:\n{fd}")
    print(f"  max |autograd - finite difference| = {(autograd - fd).abs().max().item():.2e}")

    # Mean loss: gradient equals the average of per-example gradients.
    per_example = []
    for k in range(4):
        model.zero_grad()
        bce(model(Xd[k:k + 1]), Yd[k:k + 1]).backward()
        per_example.append(model.hidden.weight.grad.clone())
    avg = torch.stack(per_example).mean(0)
    print(f"  average of the 4 per-example gradients:\n{avg}")
    print(f"  max |mean-loss grad - average of per-example grads| = {(autograd - avg).abs().max().item():.2e}")


# ---------------------------------------------------------------------------
# Task 4C: symmetry experiment
# ---------------------------------------------------------------------------
def task4c_symmetry():
    section("Task 4C: symmetry experiment, all weights and biases set to zero")
    for act in ["tanh", "sigmoid"]:
        model = XORNet(act)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        bce = nn.BCEWithLogitsLoss()
        print(f"\n  hidden activation = {act}")
        for step in range(3001):
            opt.zero_grad()
            loss = bce(model(X), Y)
            loss.backward()
            if step in (0, 1, 2, 5, 10, 100, 3000):
                W1 = model.hidden.weight.data
                g = model.hidden.weight.grad
                same = torch.allclose(W1[0], W1[1])
                print(f"  step {step:4d}  W1 row0={W1[0].tolist()}  row1={W1[1].tolist()}  "
                      f"rows identical={same}  grad row0={g[0].tolist()} row1={g[1].tolist()}  loss={loss.item():.4f}")
            opt.step()
        ok = binary_report(model)
        print(f"  all four correct: {ok}")

    # All-zero init also zeroes W(2), so every hidden gradient is exactly 0 and
    # the network sits at a saddle. To see symmetry with non-zero gradients,
    # give both hidden units the same non-zero weights instead.
    print("\n  identical non-zero init (every weight 0.5, biases 0), tanh")
    model = XORNet("tanh")
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.5 if p.dim() == 2 else 0.0)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    bce = nn.BCEWithLogitsLoss()
    for step in range(3001):
        opt.zero_grad()
        loss = bce(model(X), Y)
        loss.backward()
        if step in (0, 1, 10, 100, 3000):
            W1 = model.hidden.weight.data
            g = model.hidden.weight.grad
            print(f"  step {step:4d}  W1 row0={[round(v, 4) for v in W1[0].tolist()]}  "
                  f"row1={[round(v, 4) for v in W1[1].tolist()]}  rows identical={torch.equal(W1[0], W1[1])}  "
                  f"grad row0={[round(v, 5) for v in g[0].tolist()]} row1={[round(v, 5) for v in g[1].tolist()]}  "
                  f"loss={loss.item():.4f}")
        opt.step()
    ok = binary_report(model)
    print(f"  all four correct: {ok}")


# ---------------------------------------------------------------------------
# Task 4D: activation experiment
# ---------------------------------------------------------------------------
def task4d_activations(seed=0, n_seeds=20):
    section(f"Task 4D: activation experiment (seed={seed}, same init for every activation)")
    rows = []
    for act in ["sigmoid", "tanh", "relu"]:
        torch.manual_seed(seed)
        model = XORNet(act)
        _, final_loss, g0 = train(model, nn.BCEWithLogitsLoss(), Y, record_grad_at=0)
        torch.manual_seed(seed)
        model2 = XORNet(act)
        _, _, g10 = train(model2, nn.BCEWithLogitsLoss(), Y, record_grad_at=10)
        with torch.no_grad():
            ok = bool(((torch.sigmoid(model(X)) >= 0.5).long() == Y.long()).all())
            pre = model.hidden(X)
        rows.append((act, final_loss, ok, g0, g10))
        print(f"\n  {act}: final hidden pre-activations a(1) per input:")
        for x, r in zip(X.tolist(), pre.tolist()):
            print(f"    x={x}  a=[{r[0]: .3f}, {r[1]: .3f}]")

    print("\n  | Hidden activation | Final loss | 4/4 correct? | ||grad W1|| step 0 | ||grad W1|| step 10 |")
    print("  |---|---|---|---|---|")
    for act, fl, ok, g0, g10 in rows:
        print(f"  | {act} | {fl:.6f} | {'yes' if ok else 'no'} | {g0:.4f} | {g10:.4f} |")

    print(f"\n  Repeated-run check over seeds 0..{n_seeds - 1} (5000 Adam steps, lr=0.05):")
    print("  | Hidden activation | runs with 4/4 correct | mean final loss |")
    print("  |---|---|---|")
    for act in ["sigmoid", "tanh", "relu"]:
        wins, losses = 0, []
        for s in range(n_seeds):
            torch.manual_seed(s)
            m = XORNet(act)
            _, fl, _ = train(m, nn.BCEWithLogitsLoss(), Y)
            with torch.no_grad():
                wins += bool(((torch.sigmoid(m(X)) >= 0.5).long() == Y.long()).all())
            losses.append(fl)
        print(f"  | {act} | {wins}/{n_seeds} | {sum(losses) / len(losses):.4f} |")


# ---------------------------------------------------------------------------
# Task 5: three-class extension
# ---------------------------------------------------------------------------
def task5_multiclass(seed=0):
    section("Task 5: three-class extension (3 logits, CrossEntropyLoss)")
    torch.manual_seed(seed)
    model = XORNet("tanh", n_out=3)
    print(f"  final weight matrix W(2) shape = {tuple(model.out.weight.shape)}  (predicted (3, 2))")
    print(f"  logits per example = {model(X).shape[1]}  (predicted 3)")
    ce = nn.CrossEntropyLoss()
    init_loss, final_loss, _ = train(model, ce, Y3)
    print(f"  initial loss = {init_loss:.6f}  (ln 3 = {torch.log(torch.tensor(3.0)).item():.6f})")
    print(f"  final loss   = {final_loss:.6f}")
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
    for x, p, t in zip(X.tolist(), probs.tolist(), Y3.tolist()):
        print(f"  x={x}  P=[{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]  predicted={max(range(3), key=lambda k: p[k])}  target={t}")
    print(f"  all four correct: {bool((probs.argmax(1) == Y3).all())}")

    p = probs[1]
    print(f"\n  softmax vector for x=[0,1]: {p.tolist()}  sum = {p.sum().item():.10f}")

    # Shift invariance: add 100 to every logit.
    shifted = torch.softmax(logits[1] + 100.0, dim=0)
    print(f"  softmax(logits + 100) = {shifted.tolist()}")
    print(f"  max |difference| = {(shifted - p).abs().max().item():.2e}")
    naive = torch.exp(logits[1].double() + 1000.0)
    print(f"  naive exp(logits + 1000) in float64 = {naive.tolist()} (overflow, division gives nan)")
    stable = torch.exp(logits[1] + 1000.0 - (logits[1] + 1000.0).max())
    print(f"  after subtracting the max logit: {(stable / stable.sum()).tolist()}")

    # p - y check: gradient of CE w.r.t. logits for a single example.
    z = logits[1].clone().detach().requires_grad_(True)
    loss = nn.functional.cross_entropy(z.unsqueeze(0), Y3[1:2])
    loss.backward()
    onehot = torch.nn.functional.one_hot(Y3[1], 3).float()
    print(f"\n  dL/dz from autograd = {z.grad.tolist()}")
    print(f"  p - y               = {(torch.softmax(z, 0) - onehot).tolist()}")
    print(f"  max |difference|    = {(z.grad - (torch.softmax(z, 0) - onehot)).abs().max().item():.2e}")


if __name__ == "__main__":
    task1_linear_baseline()
    task4a_basic()
    task4b_backprop()
    task4c_symmetry()
    task4d_activations()
    task5_multiclass()
