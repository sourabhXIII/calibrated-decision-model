"""Build every blog figure as a standalone SVG, light and dark, in figures/svg/.

Data comes from the real runs in outputs/ (blog_data.json, toy_mask.json) and, for the OpenJev comparison, from the receipt in the
Verdict-open-jev repo (numbers copied below with their source).

Run from anywhere:  python make_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from figlib import Axes, Fig

HERE = Path(__file__).resolve().parent
OUT = HERE / "svg"
OUT.mkdir(exist_ok=True)
DATA = HERE.parent / "outputs"
D = json.loads((DATA / "blog_data.json").read_text())
TOY = json.loads((DATA / "toy_mask.json").read_text())

SEG_F = {0: "fc", 1: "f1", 2: "f2", 3: "f3"}       # context, Q1, Q2, Q3
SEG_W = {0: "fwc", 1: "fw1", 2: "fw2", 3: "fw3"}


def fmt_pct(v):
    return f"{v * 100:.0f}%"


# ---------------------------------------------------------------- 01 three ways
def fig_three_ways():
    f = Fig(780, 330, "Three ways to ask three questions about one document",
            "Separate calls copy the context three times. Naive concatenation reads it once "
            "but later questions see earlier ones. Branches read it once and keep questions isolated.")
    f.text(20, 26, "Three ways to ask three questions about one document", "title")
    cols = [(20, "1. Separate calls"), (280, "2. One prompt, stacked"), (540, "3. One prompt, branches")]
    for x, t in cols:
        f.text(x, 62, t, "lab", weight=600)
    cw, qw, bh = 110, 34, 22
    # 1. separate calls
    for i, cls in enumerate(["f1", "f2", "f3"]):
        y = 84 + i * 42
        f.rect(20, y, cw, bh, "fc", rx=3)
        f.text(20 + cw / 2, y + 15, "context", "small", "middle")
        f.rect(20 + cw + 2, y, qw, bh, cls, rx=3)
        f.text(20 + cw + 2 + qw / 2, y + 15, f"Q{i+1}", "small", "middle")
        f.arrow(20 + cw + qw + 8, y + bh / 2, 20 + cw + qw + 30, y + bh / 2, "lm")
        f.text(20 + cw + qw + 36, y + 15, f"A{i+1}", "small t2")
    f.text(20, 236, "context processed 3 times", "small t2")
    f.text(20, 254, "answers isolated", "small t2")
    # 2. naive stacked
    y = 126
    x = 280
    f.rect(x, y, 96, bh, "fc", rx=3)
    f.text(x + 48, y + 15, "context", "small", "middle")
    xs = []
    for i, cls in enumerate(["f1", "f2", "f3"]):
        xq = x + 98 + i * (qw + 2)
        xs.append(xq)
        f.rect(xq, y, qw, bh, cls, rx=3)
        f.text(xq + qw / 2, y + 15, f"Q{i+1}", "small", "middle")
    # leak arcs: Q2 reads Q1, Q3 reads Q1 and Q2
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        x1, x2 = xs[a] + qw / 2, xs[b] + qw / 2
        top = y - 18 - 10 * (b - a)
        f.path(f"M{x1:.1f},{y - 2} C{x1:.1f},{top} {x2:.1f},{top} {x2:.1f},{y - 4}", "thin l2")
        f.path(f"M{x2 - 4:.1f},{y - 10} L{x2:.1f},{y - 3} L{x2 + 4:.1f},{y - 10}", "thin l2")
    f.text(xs[1] + 4, y - 44, "Q2 and Q3 read the questions before them", "small t2", "middle")
    f.text(280, 236, "context processed once", "small t2")
    f.text(280, 254, "answers leak into each other", "small t2")
    # 3. branches
    x = 540
    yc = 126
    f.rect(x, yc, 96, bh, "fc", rx=3)
    f.text(x + 48, yc + 15, "context", "small", "middle")
    for i, cls in enumerate(["f1", "f2", "f3"]):
        yq = 84 + i * 42
        xq = x + 140
        f.path(f"M{x + 96:.1f},{yc + bh / 2:.1f} C{x + 118:.1f},{yc + bh / 2:.1f} "
               f"{x + 118:.1f},{yq + bh / 2:.1f} {xq - 2:.1f},{yq + bh / 2:.1f}", "thin lm")
        f.rect(xq, yq, qw, bh, cls, rx=3)
        f.text(xq + qw / 2, yq + 15, f"Q{i+1}", "small", "middle")
        f.arrow(xq + qw + 6, yq + bh / 2, xq + qw + 26, yq + bh / 2, "lm")
        f.text(xq + qw + 32, yq + 15, f"A{i+1}", "small t2")
    f.text(540, 236, "context processed once", "small t2")
    f.text(540, 254, "answers isolated", "small t2")
    f.text(20, 300, "Option 3 is what the rest of this section builds: one forward pass, no leakage, "
           "and every answer identical to option 1.", "small t2")
    f.save(OUT / "01-three-ways.svg")


# ---------------------------------------------------------------- 02 leakage
def fig_leakage():
    L = D["leakage"]
    f = Fig(780, 300, "Stacking questions changes the answers; branches do not",
            "Typed label probabilities for two questions. Isolated runs and IPPD branches agree "
            "to five decimals. Naive concatenation moves Q2 from 'no' to 'yes' and Q3 from "
            "'medium' to 'high'.")
    f.text(20, 26, "Stacking questions changes the answers. Branches don't.", "title")
    f.text(20, 46, "Qwen2.5-0.5B, probability over each question's label tokens. "
           "Isolated and branch values agree to 5 decimals, so they share one bar.", "sub")
    panels = [("Q2", "Does the customer threaten to cancel?", 20),
              ("Q3", "How urgent is this ticket?", 410)]
    for q, title, x0 in panels:
        f.text(x0, 82, f"{q}: {title}", "lab", weight=600)
        labs = L[q]["labels"]
        base, full = x0 + 70, 250
        y = 100
        for i, lab in enumerate(labs):
            yy = y + i * 52
            f.text(base - 10, yy + 21, lab, "lab", "end")
            for j, (key, cls) in enumerate([("ippd", "f1"), ("naive", "f2")]):
                v = L[q][key][i]
                by = yy + j * 18
                f.hbar(base, by, v * full, 14, cls)
                f.text(base + v * full + 6, by + 11, f"{v:.2f}", "small t2")
        f.line(base, y - 4, base, y + len(labs) * 52 - 12, "axis")
    f.legend(20, 282, [("f1", "isolated run = branch (IPPD)"), ("f2", "naive stacked prompt")])
    f.save(OUT / "02-leakage.svg")


# ---------------------------------------------------------------- 03 mask
def fig_mask():
    pos, qtn, ippd = TOY["pos_virt"], TOY["qtn"], TOY["mask"]
    n = len(qtn)
    naive = [[1 if j <= i else 0 for j in range(n)] for i in range(n)]
    labels, cnt = [], {}
    for t in qtn:
        cnt[t] = cnt.get(t, 0) + 1
        labels.append(f"c{cnt[t]}" if t == 0 else f"q{t}.{cnt[t]}")
    C, GAP, LEFT, TOP = 20, 80, 56, 118
    W = LEFT + 2 * n * C + GAP + 24
    f = Fig(W, TOP + n * C + 96, "Plain causal mask vs the IPPD branch mask",
            "Rows are queries, columns are keys. Left: a causal mask over the stacked sequence; "
            "cells marked ! are a question attending to a different question. Right: the IPPD "
            "mask, where each question block sees the shared context and its own earlier tokens only.")
    f.text(20, 26, "Who may look at whom", "title")
    f.text(20, 46, "Rows are tokens doing the looking (queries), columns are tokens being looked at "
           "(keys).", "sub")
    f.text(20, 62, "Context of 6 tokens, then questions of 3, 2 and 4.", "sub")

    def panel(mask, x0, title, positions):
        f.text(x0, TOP - 30, title, "lab", weight=600)
        for i in range(n):
            for j in range(n):
                x, y = x0 + j * C, TOP + i * C
                if mask[i][j]:
                    f.rect(x, y, C - 2, C - 2, SEG_F[qtn[j]], rx=2)
                    if qtn[i] and qtn[j] and qtn[i] != qtn[j]:
                        f.text(x + C / 2 - 1, y + C / 2 + 4, "!", "small", "middle", weight=700)
                else:
                    f.rect(x, y, C - 2, C - 2, "fb", rx=2)
        for k in range(n):
            f.text(x0 - 6, TOP + k * C + C / 2 + 3, labels[k], "tick", "end")
            f.text(x0 + k * C + C / 2 - 1, TOP + n * C + 16, str(positions[k]), "tick", "middle")
        f.text(x0 - 6, TOP + n * C + 16, "pos", "tick", "end")

    panel(naive, LEFT, "Plain causal mask", list(range(n)))
    panel(ippd, LEFT + n * C + GAP, "IPPD branch mask", pos)
    ly = TOP + n * C + 50
    f.legend(20, ly, [("fc", "shared context"), ("f1", "question 1"), ("f2", "question 2"),
                      ("f3", "question 3"), ("fb", "blocked")])
    f.text(20, ly + 24, "! = a question reading another question.   pos = the position ID each "
           "token is given. On the right, every question restarts at 6.", "small t2")
    f.save(OUT / "03-mask.svg")


# ---------------------------------------------------------------- 04 positions
def fig_positions():
    pos, qtn = TOY["pos_virt"], TOY["qtn"]
    n = len(qtn)
    C, X0 = 40, 150
    f = Fig(X0 + n * C + 30, 262, "Physical order vs virtual positions",
            "The same 15 tokens. Their physical index runs 0 to 14. Their virtual position IDs "
            "restart at 6 for each question, which is where each question would sit if asked alone.")
    f.text(20, 26, "Same tokens, two numberings", "title")
    f.text(20, 46, "Where a token sits in memory vs the position number the model is told.", "sub")
    for row, (lab, vals, y) in enumerate([("memory index", list(range(n)), 80),
                                          ("position ID", pos, 150)]):
        f.text(X0 - 12, y + 25, lab, "lab", "end")
        for k in range(n):
            f.rect(X0 + k * C, y, C - 3, 38, SEG_F[qtn[k]], rx=3)
            f.text(X0 + k * C + (C - 3) / 2, y + 24, str(vals[k]), "lab", "middle", weight=600)
    # brackets under each question in the position row
    spans = {}
    for k, t in enumerate(qtn):
        spans.setdefault(t, [k, k])[1] = k
    for t in (1, 2, 3):
        a, b = spans[t]
        xa, xb = X0 + a * C + 2, X0 + (b + 1) * C - 5
        f.path(f"M{xa:.1f},196 V202 H{xb:.1f} V196", "thin lm")
    f.text(X0 + 6 * C, 224, "each question restarts at 6, right after the context,", "small t2")
    f.text(X0 + 6 * C, 240, "as if it were the only question", "small t2")
    f.save(OUT / "04-positions.svg")


# ---------------------------------------------------------------- 05 cost
def fig_cost():
    rows = D["cost"]
    f = Fig(780, 360, "Prefill time vs number of questions",
            "Wall-clock prefill for single-token answers on one GPU, separate calls vs one stacked "
            "sequence with branches vs naive stacking. Left: 125-token context. Right: 1,241-token "
            "context. Separate calls grow linearly; branches stay nearly flat.")
    f.text(20, 26, "More questions, almost no extra time", "title")
    f.text(20, 46, "Prefill time for single-token answers, Qwen2.5-0.5B bf16 on an NVIDIA GB10, "
           "log scale. Separate calls run as one", "sub")
    f.text(20, 62, "padded batch with no prefix cache.", "sub")
    panels = [(125, 70, "125-token context"), (1241, 450, "1,241-token context")]
    for ctx, x0, title in panels:
        rs = [r for r in rows if r["ctx"] == ctx]
        ax = Axes(f, x0, 100, 210, 190, (1, 64), (5, 3000), xlog=True, ylog=True)
        f.text(x0, 90, title, "lab", weight=600)
        ax.grid_y([10, 30, 100, 300, 1000, 3000], lambda v: f"{v:,}",
                  "prefill ms" if x0 < 100 else None)
        ax.ticks_x([1, 2, 4, 8, 16, 32, 64], str, "questions per context")
        ax.series([(r["M"], r["naive_ms"]) for r in rs], "l3")
        ax.series([(r["M"], r["ippd_ms"]) for r in rs], "l1")
        ax.series([(r["M"], r["sep_ms"]) for r in rs], "l2")
        last = rs[-1]
        lx = ax.px(64) + 10
        f.text(lx, ax.py(last["sep_ms"]) + 4, f"separate {last['sep_ms']:,.0f}", "small")
        f.text(lx, ax.py(last["ippd_ms"]) - 3, f"branches {last['ippd_ms']:.0f}", "small")
        f.text(lx, ax.py(last["naive_ms"]) + 13, f"naive {last['naive_ms']:.0f}", "small")
    f.legend(70, 348, [("f2", "separate calls"), ("f1", "branches (IPPD)"),
                       ("f3", "naive stacking (leaks)")])
    f.save(OUT / "05-cost.svg")


# ---------------------------------------------------------------- 06 token mass
def fig_token_mass():
    toks = D["token_mass"]["Q2"][:8]
    f = Fig(780, 330, "Where the probability mass sits for Q2",
            "Next-token probabilities at Q2's answer slot. ' No' holds 0.79 and ' Yes' 0.17; the "
            "label tokens ' no' and ' yes' hold 0.012 and 0.004. Renormalising over the label "
            "tokens alone gives no = 0.76 from 1.5% of the mass.")
    f.text(20, 26, "The label tokens hold 1.5% of the probability", "title")
    f.text(20, 46, "Q2 \"Does the customer threaten to cancel?\": top next tokens at the answer slot.",
           "sub")
    base, full, y0 = 120, 420, 70
    for i, t in enumerate(toks):
        tokstr = t["token"].replace(" ", "␣").replace("\n", "\\n")
        y = y0 + i * 26
        is_label = t["token"] in (" yes", " no")
        is_variant = t["token"] in (" Yes", " No")
        cls = "f1" if is_label else ("f2" if is_variant else "fc")
        f.text(base - 10, y + 13, tokstr, "lab", "end")
        f.hbar(base, y, max(t["p"] * full, 1.5), 16, cls)
        f.text(base + t["p"] * full + 6, y + 12, f"{t['p']:.3f}", "small t2")
    f.line(base, y0 - 4, base, y0 + len(toks) * 26 - 6, "axis")
    m = {t["token"]: t["p"] for t in D["token_mass"]["Q2"]}
    pn, py_ = m[" no"], m[" yes"]
    pN, pY = m[" No"], m[" Yes"]
    lx = 590
    f.text(lx, 96, "Read only the label tokens", "lab", weight=600)
    f.text(lx, 114, "and renormalise:", "small t2")
    f.text(lx, 134, f"no = {pn:.4f} / ({pn:.4f} + {py_:.4f})", "small")
    f.text(lx, 152, f"   = {pn / (pn + py_):.2f}, from {(pn + py_) * 100:.1f}% of the mass", "small")
    f.text(lx, 196, "Pool case variants:", "lab", weight=600)
    f.text(lx, 216, f"no = {(pn + pN) / (pn + py_ + pN + pY):.2f}, from "
           f"{(pn + py_ + pN + pY) * 100:.1f}% of the mass", "small")
    f.legend(20, 300, [("f1", "label tokens we read"), ("f2", "same answers, other tokens"),
                       ("fc", "everything else")])
    f.text(20, 320, "␣ marks a leading space. Exact probabilities from the isolated run, "
           "Qwen2.5-0.5B-Instruct fp32.", "small tm")
    f.save(OUT / "06-token-mass.svg")


# ---------------------------------------------------------------- 07 proper scores
def fig_proper():
    p = 0.7
    f = Fig(780, 344, "Proper scoring rules reward reporting the truth",
            "Expected log loss and Brier score when the true chance of rain is 70%, as a function "
            "of the reported probability. Both are lowest at exactly 70%. Exaggerating to 90% costs "
            "0.765 vs 0.611 in log loss and 0.250 vs 0.210 in Brier.")
    f.text(20, 26, "Lying costs you, in expectation", "title")
    f.text(20, 46, "The true chance of rain is 70%. Expected penalty for each forecast you could "
           "announce (lower is better).", "sub")
    qs = [i / 100 for i in range(6, 99)]
    funcs = [("Log loss", lambda q: -(p * math.log(q) + (1 - p) * math.log(1 - q)),
              [0, 0.5, 1.0, 1.5, 2.0], 70),
             ("Brier score", lambda q: p * (1 - q) ** 2 + (1 - p) * q ** 2,
              [0, 0.2, 0.4, 0.6, 0.8], 450)]
    for name, fn, yt, x0 in funcs:
        ax = Axes(f, x0, 90, 260, 180, (0, 1), (0, yt[-1]))
        f.text(x0, 80, name, "lab", weight=600)
        ax.grid_y(yt, lambda v: f"{v:g}", "expected penalty" if x0 < 100 else None)
        ax.ticks_x([0, 0.25, 0.5, 0.75, 1.0], lambda v: f"{v:g}", "announced probability of rain")
        pts = [(q, min(fn(q), yt[-1])) for q in qs]
        ax.series(pts, "l1", dots=False)
        for q, cls, lab, dx, dy, anc in [(0.7, "f1", "honest 70%", -4, 24, "end"),
                                         (0.9, "f2", "exaggerate 90%", -10, -16, "end")]:
            v = fn(q)
            f.dot(ax.px(q), ax.py(v), cls, r=5)
            f.text(ax.px(q) + dx, ax.py(v) + dy, f"{lab}: {v:.3f}", "small", anc)
    f.text(20, 334, "Same picture for any true probability: the curve bottoms out exactly at the truth. "
           "That is what \"proper\" means.", "small t2")
    f.save(OUT / "07-proper-scores.svg")


# ---------------------------------------------------------------- 08 reliability
def fig_reliability():
    C = D["calibration"]
    f = Fig(780, 446, "Reliability diagram for the toy task",
            "Confidence vs accuracy in 10 bins. The supervised and RL students sit on the diagonal. "
            "The student distilled from an overconfident teacher sits below it: when it says 90% it "
            "is right about 75% of the time.")
    f.text(20, 26, "Copy an overconfident teacher, get an overconfident student", "title")
    f.text(20, 46, "Toy task with known true probabilities. Each point: one confidence bin, "
           "20,000 test inputs.", "sub")
    ax = Axes(f, 80, 80, 300, 300, (0.2, 1.0), (0.2, 1.0))
    ax.grid_y([0.2, 0.4, 0.6, 0.8, 1.0], lambda v: f"{v:.1f}", "accuracy in bin")
    ax.ticks_x([0.2, 0.4, 0.6, 0.8, 1.0], lambda v: f"{v:.1f}", "stated confidence")
    f.line(ax.px(0.2), ax.py(0.2), ax.px(1.0), ax.py(1.0), "thin lm")
    f.text(ax.px(0.97), ax.py(0.99) - 6, "perfect", "small tm", "end")
    for key, cls in [("kd_overconf", "l2"), ("rl_gauss", "l3"), ("sl_hard", "l1")]:
        ax.series([(r["conf"], r["acc"]) for r in C[key]["reliability"]], cls)
    r = C["kd_overconf"]["reliability"][-2]
    f.text(ax.px(r["conf"]) + 6, ax.py(r["acc"] - 0.09),
           f"says {r['conf']:.0%},", "small t2")
    f.text(ax.px(r["conf"]) + 6, ax.py(r["acc"] - 0.09) + 15,
           f"right {r['acc']:.0%}", "small t2")
    tx = 470
    f.text(tx, 100, "Final test numbers", "lab", weight=600)
    rows = [("", "ECE", "KL to truth"),
            ("supervised (log loss)", C["sl_hard"]["final"]["ece"], C["sl_hard"]["final"]["kl_to_truth"]),
            ("RL, proper-score reward", C["rl_gauss"]["final"]["ece"], C["rl_gauss"]["final"]["kl_to_truth"]),
            ("distil, honest teacher", C["kd_true"]["final"]["ece"], max(C["kd_true"]["final"]["kl_to_truth"], 0)),
            ("distil, overconfident", C["kd_overconf"]["final"]["ece"], C["kd_overconf"]["final"]["kl_to_truth"])]
    for i, (n, a, b) in enumerate(rows):
        y = 126 + i * 24
        f.text(tx, y, n, "small" if i else "small tm")
        f.text(tx + 200, y, a if i == 0 else f"{a:.3f}", "small tick" if i == 0 else "small", "end")
        f.text(tx + 285, y, b if i == 0 else f"{b:.3f}", "small tick" if i == 0 else "small", "end")
    f.text(tx, 270, "The honest-teacher student lands exactly", "small t2")
    f.text(tx, 286, "on the oracle, so it is not drawn.", "small t2")
    f.legend(80, 434, [("f1", "supervised"), ("f3", "RL"), ("f2", "student of overconfident teacher")])
    f.save(OUT / "08-reliability.svg")


# ---------------------------------------------------------------- 09 gradient cosine
def fig_cosine():
    cos = D["grad_cosine"]
    lo, hi, w = 0.88, 1.0, 0.005
    nb = round((hi - lo) / w)
    counts = [0] * nb
    for c in cos:
        counts[min(nb - 1, max(0, int((c - lo) / w)))] += 1
    mean = sum(cos) / len(cos)
    f = Fig(780, 320, "RL gradient direction vs supervised gradient direction",
            f"Histogram of the cosine similarity between the RL gradient estimate and the exact "
            f"supervised log-loss gradient, over 200 batches. Mean {mean:.2f}, minimum {min(cos):.2f}.")
    f.text(20, 26, "The RL update points where supervised learning points", "title")
    f.text(20, 46, "Cosine similarity between the Gaussian-noise RL gradient and the log-loss "
           "gradient, 200 batches of 256 (1.0 = same direction).", "sub")
    ymax = 50
    ax = Axes(f, 80, 80, 640, 180, (lo, hi), (0, ymax))
    ax.grid_y([0, 10, 20, 30, 40, 50], str, "batches")
    ax.ticks_x([0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0], lambda v: f"{v:.2f}", "cosine similarity")
    bw = ax.px(lo + w) - ax.px(lo)
    for i, c in enumerate(counts):
        x = ax.px(lo + i * w) + 1
        f.vbar(x, ax.py(0), ax.py(0) - ax.py(min(c, ymax)), bw - 2, "f1")
    f.line(ax.px(mean), 80, ax.px(mean), ax.py(0), "thin lk")
    f.text(ax.px(mean) - 6, 92, f"mean {mean:.2f}", "small", "end")
    f.save(OUT / "09-grad-cosine.svg")


# ---------------------------------------------------------------- 10 gliclass
def fig_gliclass():
    f = Fig(780, 364, "How a GLiClass-style encoder scores labels",
            "Labels and text go into one sequence with marker tokens. A bidirectional encoder reads "
            "everything. The hidden state at each label marker is scored against a pooled text "
            "vector, and a softmax over labels gives the answer distribution.")
    f.text(20, 26, "Labels are part of the input, so they can change on every request", "title")
    toks = [("<<LABEL>>", "fc"), ("billing", "f1"), ("<<LABEL>>", "fc"), ("technical", "f2"),
            ("<<LABEL>>", "fc"), ("other", "f3"), ("<<SEP>>", "fc"), ("We", "fb"), ("were", "fb"),
            ("billed", "fb"), ("twice", "fb"), ("…", "fb")]
    x, y = 20, 296
    pos = []
    for t, cls in toks:
        w = max(34, 6.3 * len(t) + 14)
        f.rect(x, y, w, 26, cls, rx=3)
        f.text(x + w / 2, y + 17, t, "small", "middle")
        pos.append((x, w))
        x += w + 4
    right = x - 4
    f.rect(20, 214, right - 20, 46, "fwc", rx=6)
    f.text(20 + (right - 20) / 2, 241, "bidirectional encoder: every token attends to every token",
           "lab", "middle")
    for px_, w in pos:
        f.line(px_ + w / 2, y - 2, px_ + w / 2, 262, "thin lm")
    # one vector per label marker, one pooled text vector
    vec_y = 160
    centres = []
    for idx, cls in [(0, "f1"), (2, "f2"), (4, "f3")]:
        cx = pos[idx][0] + pos[idx][1] / 2
        f.line(cx, 212, cx, vec_y + 22, "thin lm")
        f.rect(cx - 9, vec_y, 18, 22, cls, rx=3)
        centres.append(cx)
    tx0 = pos[7][0]
    tx1 = pos[-1][0] + pos[-1][1]
    tc = (tx0 + tx1) / 2
    f.path(f"M{tx0 + 4:.1f},212 Q{tc:.1f},186 {tx1 - 4:.1f},212", "thin lm")
    f.line(tc, 199, tc, vec_y + 22, "thin lm")
    f.rect(tc - 9, vec_y, 18, 22, "fc", rx=3)
    f.text(centres[0] - 12, vec_y + 15, "label vectors", "small t2", "end") if centres[0] > 120 else None
    f.text(tc + 14, vec_y + 15, "pooled text", "small t2")
    # scorer box
    bx0, bx1, by0, by1 = 20, right, 56, 126
    f.rect(bx0, by0, bx1 - bx0, by1 - by0, "fwc", rx=6)
    f.text(bx0 + 14, by0 + 26, "score each (pooled text, label vector) pair", "lab", weight=600)
    f.text(bx0 + 14, by0 + 46, "then softmax over the labels in this request", "small t2")
    for cx in centres + [tc]:
        f.arrow(cx, vec_y - 2, cx, by1 + 3, "lm")
    probs = [("billing", 0.86, "f1"), ("technical", 0.05, "f2"), ("other", 0.09, "f3")]
    for i, (lab, v, cls) in enumerate(probs):
        yy = by0 + 12 + i * 18
        f.text(bx1 - 190, yy + 10, lab, "small", "end")
        f.hbar(bx1 - 184, yy, v * 130, 12, cls)
        f.text(bx1 - 184 + v * 130 + 6, yy + 10, f"{v:.2f}", "small t2")
    f.text(20, 350, "Illustrative probabilities. Input format from Knowledgator/GLiClass "
           "(uni-encoder): <<LABEL>>label ... <<SEP>>text.", "small tm")
    f.save(OUT / "10-gliclass.svg")


# ---------------------------------------------------------------- 11 layouts
def fig_layouts():
    f = Fig(780, 330, "Encoder rows vs decoder branches",
            "Top: open encoder replicas put each question in its own row with a full copy of the "
            "state, so the state is encoded once per question. Bottom: a decoder with branch masks "
            "encodes the state once and hangs every question off it.")
    f.text(20, 26, "Where the state gets read", "title")
    f.text(20, 62, "Encoder, one row per question (OpenJev, Laya as published)", "lab", weight=600)
    for i, cls in enumerate(["f1", "f2", "f3"]):
        y = 76 + i * 30
        f.rect(20, y, 90, 24, cls, rx=3)
        f.text(65, y + 16, f"Q{i+1} + options", "small", "middle")
        f.rect(112, y, 300, 24, "fc", rx=3)
        f.text(262, y + 16, "state (full copy)", "small", "middle")
    f.text(430, 112, "state encoded 3 times", "small t2")
    f.text(430, 130, "questions isolated: yes", "small t2")
    f.text(430, 148, "options see each other: yes", "small t2")
    f.text(20, 202, "Decoder with branch masks (Block 1)", "lab", weight=600)
    y = 216
    f.rect(20, y, 300, 24, "fc", rx=3)
    f.text(170, y + 16, "state (once)", "small", "middle")
    for i, cls in enumerate(["f1", "f2", "f3"]):
        xq = 322 + i * 94
        f.rect(xq, y, 92, 24, cls, rx=3)
        f.text(xq + 46, y + 16, f"Q{i+1} + options", "small", "middle")
    f.text(620, 222, "state encoded once", "small t2")
    f.text(620, 240, "questions isolated: yes", "small t2")
    f.text(620, 258, "options read in order", "small t2")
    f.text(20, 306, "Both keep questions isolated. Only the branch layout shares the state, which is "
           "where the 20× prefill saving comes from.", "small t2")
    f.save(OUT / "11-layouts.svg")


# ---------------------------------------------------------------- 12 eval by type
def fig_eval():
    # reports/v2/exp_e9_external_cases.json in Heman10x-NGU/Verdict-open-jev @ 30f1556
    rows = [("yes/no (noul)", 220, 0.6136, 0.9318), ("choice", 90, 0.2556, 0.9000),
            ("score", 27, 0.1481, 0.7407), ("all", 337, 0.4807, 0.9080)]
    f = Fig(780, 344, "OpenJev vs Jev on TypeSafe's public workflow cases",
            "Agreement with TypeSafe's reference answers (average of GPT-6 Astra and Claude Fable "
            "5.1) by question type. OpenJev 61%, 26%, 15%, 48% overall; Jev 93%, 90%, 74%, 91% "
            "overall.")
    f.text(20, 26, "48% vs 91%, and what it measures", "title")
    f.text(20, 46, "Share of questions where the top answer matches TypeSafe's reference: the average "
           "of GPT-6 Astra and", "sub")
    f.text(20, 62, "Claude Fable 5.1 answers. Not ground truth.", "sub")
    base, full, y0 = 170, 460, 86
    for i, (lab, n, ours, jev) in enumerate(rows):
        y = y0 + i * 52
        f.text(base - 10, y + 16, lab, "lab", "end")
        f.text(base - 10, y + 31, f"n = {n}", "small tm", "end")
        for j, (v, cls) in enumerate([(jev, "f1"), (ours, "f2")]):
            by = y + j * 18
            f.hbar(base, by, v * full, 14, cls)
            f.text(base + v * full + 6, by + 11, fmt_pct(v), "small t2")
    f.line(base, y0 - 4, base, y0 + 4 * 52 - 14, "axis")
    f.legend(20, 314, [("f1", "Jev (published answers)"),
                       ("f2", "OpenJev 151M, fine-tuned on Banking77, docs cut to 1,200 chars")])
    f.text(20, 336, "Source: reports/v2/exp_e9_external_cases.json, Heman10x-NGU/Verdict-open-jev.",
           "small tm")
    f.save(OUT / "12-eval-by-type.svg")


# ---------------------------------------------------------------- 13 sketch
def fig_sketch():
    f = Fig(780, 370, "A plausible System One pipeline, labelled by evidence",
            "State is read once; questions become isolated branches; a trained head scores option "
            "slots; typed answers and probabilities come out. Each stage is tagged documented, "
            "inferred or guess.")
    f.text(20, 26, "A plausible pipeline, with every box tagged by how much we know", "title")
    boxes = [
        (20, "state", "read once", "documented"),
        (172, "question branches", "isolated, parallel", "documented"),
        (324, "branch masks", "shared prefix", "inferred"),
        (476, "option-slot head", "no vocab readout", "inferred"),
        (628, "typed answers", "probs + confidence", "documented"),
    ]
    for x, title, sub, tag in boxes:
        f.rect(x, 90, 140, 90, "fwc", rx=8)
        f.text(x + 70, 124, title, "small", "middle", weight=600)
        f.text(x + 70, 144, sub, "small t2", "middle")
        f.rect(x + 30, 156, 80, 18, "fs", rx=9)
        f.add(f'<rect x="{x + 30}" y="156" width="80" height="18" rx="9" class="box"/>')
        f.text(x + 70, 169, tag, "small", "middle")
    for x in (160, 312, 464, 616):
        f.arrow(x + 1, 135, x + 11, 135, "lk")
    f.text(20, 220, "Training (below the line is mostly guesswork)", "lab", weight=600)
    f.line(20, 230, 760, 230, "axis")
    train = [(20, "synthetic states + typed questions", "\"we make all the data ourselves\"",
              "documented"),
             (270, "labels: frontier-model probabilities", "proper-score loss", "inferred"),
             (520, "RLCD on top", "on-policy or outcome RL?", "guess")]
    for x, title, sub, tag in train:
        f.rect(x, 246, 230, 70, "fwc", rx=8)
        f.text(x + 12, 270, title, "small", weight=600)
        f.text(x + 12, 288, sub, "small t2")
        f.add(f'<rect x="{x + 12}" y="296" width="80" height="16" rx="8" class="box"/>')
        f.text(x + 52, 308, tag, "small", "middle")
    f.text(20, 340, "documented = TypeSafe's docs or launch post say so.   inferred = follows from "
           "documented limits and behaviour.", "small tm")
    f.text(20, 356, "guess = consistent with the evidence, not implied by it.", "small tm")
    f.save(OUT / "13-sketch.svg")


# ---------------------------------------------------------------- 14 joint mosaic
def fig_joint():
    f = Fig(780, 340, "Calibrated marginals, wrong joint",
            "Two questions, A (threatens to cancel, 70%) and B (likely to churn, 60%). Multiplying "
            "isolated answers implies both are true 42% of the time. If they are correlated, the "
            "true joint can be 58%. Both mosaics have identical marginals.")
    f.text(20, 26, "Each answer calibrated, the combination wrong", "title")
    f.text(20, 46, "A = \"threatens to cancel\" (70%), B = \"likely to churn\" (60%). "
           "Same marginals in both squares.", "sub")

    def mosaic(x0, title, cells):
        S = 200
        y0 = 90
        f.text(x0, 80, title, "lab", weight=600)
        # columns: A (0.7) and not-A (0.3); cells give P(B|col) heights
        colw = {"A": 0.7, "notA": 0.3}
        xx = x0
        for col in ("A", "notA"):
            w = colw[col] * S
            pb = cells[col] / colw[col]
            hb = pb * S
            both = col == "A"
            f.rect(xx + 1, y0 + 1, w - 2, hb - 2, "f1" if both else "fw1", rx=2)
            f.rect(xx + 1, y0 + hb + 1, w - 2, S - hb - 2, "fb", rx=2)
            if both:
                f.text(xx + w / 2, y0 + hb / 2 + 5, f"A and B: {cells[col]:.2f}", "lab", "middle",
                       weight=600)
            xx += w
        f.text(x0 + 0.35 * S, y0 + S + 18, "A (0.70)", "small t2", "middle")
        f.text(x0 + 0.85 * S, y0 + S + 18, "not A", "small t2", "middle")

    mosaic(40, "Multiply the isolated answers", {"A": 0.42, "notA": 0.18})
    mosaic(300, "What is actually true (correlated)", {"A": 0.58, "notA": 0.02})
    tx = 540
    f.text(tx, 110, "Rule: escalate if A and B", "lab", weight=600)
    f.text(tx, 134, "code computes  0.70 × 0.60 = 0.42", "small")
    f.text(tx, 154, "true rate                     = 0.58", "small")
    f.text(tx, 186, "Every marginal check passes.", "small t2")
    f.text(tx, 204, "The joint is off by 16 points,", "small t2")
    f.text(tx, 222, "and no single-question loss", "small t2")
    f.text(tx, 240, "can see it.", "small t2")
    f.text(20, 326, "Shaded blue: B true. Filled blue: A and B both true. Column widths are P(A); "
           "heights within a column are P(B | column).", "small tm")
    f.save(OUT / "14-joint.svg")


# ---------------------------------------------------------------- 15 implication
def fig_implication():
    f = Fig(780, 230, "An incoherent pair of isolated answers",
            "A implies B, so P(B) must be at least P(A). Isolated branches can output P(A) = 0.8 "
            "and P(B) = 0.6 for the same invoice.")
    f.text(20, 26, "When one question implies another", "title")
    f.text(20, 46, "A = \"invoice exceeds the PO by more than 10%\"   B = \"invoice exceeds the PO\"   "
           "A implies B, so P(B) ≥ P(A).", "sub")
    ax = Axes(f, 60, 110, 660, 10, (0, 1), (0, 1))
    f.line(60, 115, 720, 115, "axis")
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        f.line(ax.px(v), 111, ax.px(v), 119, "axis")
        f.text(ax.px(v), 138, f"{v:g}", "tick", "middle")
    f.rect(ax.px(0.8), 103, ax.px(1.0) - ax.px(0.8), 24, "fw1", rx=3)
    f.text(ax.px(0.9), 96, "where P(B) must be", "small t2", "middle")
    f.dot(ax.px(0.8), 115, "f1", r=6)
    f.text(ax.px(0.8), 160, "P(A) = 0.80", "small", "middle")
    f.dot(ax.px(0.6), 115, "f2", r=6)
    f.text(ax.px(0.6), 160, "P(B) = 0.60", "small", "middle")
    f.arrow(ax.px(0.62), 180, ax.px(0.79), 180, "l2")
    f.text(ax.px(0.705), 200, "violation", "small t2", "middle")
    f.text(20, 222, "Each number can still be calibrated on average across many invoices. "
           "Calibration is a property of groups, coherence is a property of each input.", "small tm")
    f.save(OUT / "15-implication.svg")


if __name__ == "__main__":
    for fn in [fig_three_ways, fig_leakage, fig_mask, fig_positions, fig_cost, fig_token_mass,
               fig_proper, fig_reliability, fig_cosine, fig_gliclass, fig_layouts, fig_eval,
               fig_sketch, fig_joint, fig_implication]:
        fn()
        print("ok", fn.__name__)
