"""Supervised proper scoring vs RL with a proper-score reward, on a toy task
where the true probabilities are known.

Setup. Inputs x in R^8, K = 3 answers. The true answer distribution is
p*(y|x) = softmax(W* x), and labels are *sampled* from it, so even a perfect
model is uncertain (aleatoric noise). A linear student is trained five ways:

  sl_hard      : cross-entropy on sampled hard labels (log score)
  sl_brier     : multiclass Brier on hard labels
  kd_true      : cross-entropy on soft labels from a perfectly calibrated teacher p*
  kd_overconf  : soft labels from an overconfident teacher softmax(2 * W* x)
  rl_gauss     : Laya-style "RLCD": Gaussian noise on the logits, reward = log
                 score of the noisy distribution at the label, group-normalised
                 advantage, REINFORCE through the Gaussian log-density
                 (github.com/NandhaKishorM/laya, notebooks/..._mps.py)

Reported on held-out data: accuracy, NLL against sampled labels, top-label
ECE, and KL(p* || model), the distance to the truth that calibration is
ultimately about.

Run: python calib_demo.py   (CPU is fine; a few seconds per method)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

torch.manual_seed(0)
D, K, N_TRAIN, N_TEST = 8, 3, 20_000, 20_000
STEPS, LR, BATCH = 3_000, 0.05, 256

W_true = torch.randn(K, D) * 0.8


def sample(n):
    x = torch.randn(n, D)
    p = F.softmax(x @ W_true.T, -1)
    y = torch.multinomial(p, 1).squeeze(-1)
    return x, y, p


x_tr, y_tr, p_tr = sample(N_TRAIN)
x_te, y_te, p_te = sample(N_TEST)


def ece(probs, y, bins=15):
    conf, pred = probs.max(-1)
    correct = (pred == y).float()
    edges = torch.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.float().mean() * (conf[m].mean() - correct[m].mean()).abs()
    return float(total)


def evaluate(W):
    with torch.no_grad():
        logits = x_te @ W.T
        q = F.softmax(logits, -1)
        return dict(
            acc=float((q.argmax(-1) == y_te).float().mean()),
            nll=float(F.cross_entropy(logits, y_te)),
            ece=ece(q, y_te),
            kl_to_truth=float((p_te * (p_te.log() - q.log())).sum(-1).mean()),
        )


def train(loss_fn, steps=STEPS, lr=LR):
    W = torch.zeros(K, D, requires_grad=True)
    opt = torch.optim.Adam([W], lr=lr)
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        idx = torch.randint(0, N_TRAIN, (BATCH,), generator=g)
        loss = loss_fn(x_tr[idx] @ W.T, idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return W.detach()


def sl_hard(logits, idx):
    return F.cross_entropy(logits, y_tr[idx])


def sl_brier(logits, idx):
    q = F.softmax(logits, -1)
    return ((q - F.one_hot(y_tr[idx], K)) ** 2).sum(-1).mean()


def kd(teacher_scale):
    def loss(logits, idx):
        t = F.softmax(teacher_scale * (x_tr[idx] @ W_true.T), -1)
        return -(t * F.log_softmax(logits, -1)).sum(-1).mean()
    return loss


def rl_gauss(sigma=0.25, samples=4):
    """Gaussian-perturbation policy gradient with a log-score reward."""
    def loss(logits, idx):
        y = y_tr[idx]
        eps = torch.randn((samples,) + logits.shape) * sigma
        noisy = logits.detach().unsqueeze(0) + eps
        with torch.no_grad():
            reward = F.log_softmax(noisy, -1).gather(-1, y.expand(samples, -1).unsqueeze(-1)).squeeze(-1)
            adv = reward - reward.mean(0, keepdim=True)
            adv = adv / (adv.std() + 1e-6)
        logp = -((noisy - logits.unsqueeze(0)) ** 2).sum(-1) / (2 * sigma ** 2)
        return -(adv * logp).mean()
    return loss


def grad_cosine(n_batches=50, sigma=0.25):
    """Cosine between the RL gradient estimate and the exact SL gradient at a
    random point. Shows how noisy the RL estimator is per batch."""
    W = (torch.randn(K, D) * 0.3).requires_grad_(True)
    cos = []
    for b in range(n_batches):
        idx = torch.randint(0, N_TRAIN, (BATCH,))
        g_sl = torch.autograd.grad(sl_hard(x_tr[idx] @ W.T, idx), W)[0].flatten()
        g_rl = torch.autograd.grad(rl_gauss(sigma)(x_tr[idx] @ W.T, idx), W)[0].flatten()
        cos.append(F.cosine_similarity(g_sl, g_rl, dim=0))
    c = torch.stack(cos)
    return float(c.mean()), float(c.std())


if __name__ == "__main__":
    print(f"torch {torch.__version__} | D={D} K={K} train={N_TRAIN} test={N_TEST} "
          f"steps={STEPS} batch={BATCH}\n")
    oracle = evaluate(W_true)
    print(f"{'oracle (true W*)':<16} acc {oracle['acc']:.3f}  nll {oracle['nll']:.3f}  "
          f"ece {oracle['ece']:.3f}  KL {oracle['kl_to_truth']:.4f}")
    runs = [("sl_hard", sl_hard), ("sl_brier", sl_brier), ("kd_true", kd(1.0)),
            ("kd_overconf", kd(2.0)), ("rl_gauss", rl_gauss())]
    for name, fn in runs:
        r = evaluate(train(fn))
        print(f"{name:<16} acc {r['acc']:.3f}  nll {r['nll']:.3f}  ece {r['ece']:.3f}  "
              f"KL {r['kl_to_truth']:.4f}")
    m, s = grad_cosine()
    print(f"\nper-batch cosine(RL grad, SL grad) = {m:.2f} ± {s:.2f}  (1.0 = same direction)")
    # same budget of *gradient steps* is generous to RL: give it 10x and compare
    r = evaluate(train(rl_gauss(), steps=STEPS * 10))
    print(f"{'rl_gauss x10':<16} acc {r['acc']:.3f}  nll {r['nll']:.3f}  ece {r['ece']:.3f}  "
          f"KL {r['kl_to_truth']:.4f}")
