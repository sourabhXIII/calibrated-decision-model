"""Collect the extra data the blog figures need. Writes outputs/blog_data.json.

  token_mass   : top next-token probabilities at Q2 / Q3's answer slot (isolated
                 runs), to draw where the probability mass actually sits
  leakage      : typed label probabilities for each question under isolated,
                 IPPD and naive-concatenation runs
  cost         : prefill ms and peak memory vs number of questions, two
                 context lengths, separate calls vs IPPD vs naive
  calibration  : training curves (KL to truth vs step) and reliability bins for
                 the toy task in calib_demo.py

Run (needs a CUDA GPU):  python blog_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import calib_demo as cd
from cost_bench import QUESTION, TICKET, timed
from ippd_branches import (CONTEXT, MODEL, QUESTIONS, build_stacked, forward_logits,
                           ippd_mask, to_additive)

out = {"versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                    "gpu": torch.cuda.get_device_name(0), "model": MODEL}}

# ---------------------------------------------------------------- LLM parts
tok = AutoTokenizer.from_pretrained(MODEL)
model32 = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float32, attn_implementation="eager").to("cuda").eval()

ctx_ids = tok(CONTEXT, add_special_tokens=False)["input_ids"]
q_ids = [tok(q["text"], add_special_tokens=False)["input_ids"] for q in QUESTIONS]
ids, pos, qtn, last = build_stacked(ctx_ids, q_ids)
ippd = forward_logits(model32, ids, pos, ippd_mask(pos, qtn))
naive = forward_logits(model32, ids)
iso = [forward_logits(model32, ctx_ids + q)[-1] for q in q_ids]

out["token_mass"], out["leakage"] = {}, {}
for k, q in enumerate(QUESTIONS):
    p = torch.softmax(iso[k], -1)
    top = torch.topk(p, 12)
    out["token_mass"][f"Q{k+1}"] = [
        {"token": tok.decode([i]), "p": float(v)} for v, i in zip(top.values, top.indices)]
    lab = [tok(l, add_special_tokens=False)["input_ids"][0] for l in q["labels"]]
    out["leakage"][f"Q{k+1}"] = {
        "labels": [l.strip() for l in q["labels"]],
        "isolated": torch.softmax(iso[k][lab], -1).tolist(),
        "ippd": torch.softmax(ippd[last[k], lab], -1).tolist(),
        "naive": torch.softmax(naive[last[k], lab], -1).tolist(),
    }
del model32

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
dev = model.device
out["cost"] = []
for ctx_reps in (4, 40):
    c_ids = tok(TICKET * ctx_reps + "\n\n", add_special_tokens=False)["input_ids"]
    for M in (1, 2, 4, 8, 16, 32, 64):
        qs = [tok(QUESTION.format(i=i), add_special_tokens=False)["input_ids"] for i in range(M)]
        seqs = [c_ids + q for q in qs]
        Lmax = max(map(len, seqs))
        batch = torch.full((M, Lmax), tok.pad_token_id, device=dev)
        attn = torch.zeros((M, Lmax), dtype=torch.long, device=dev)
        for r, s in enumerate(seqs):
            batch[r, :len(s)] = torch.tensor(s)
            attn[r, :len(s)] = 1
        rows = torch.arange(M, device=dev)
        ends = torch.tensor([len(s) - 1 for s in seqs], device=dev)
        sep = lambda: model.lm_head(
            model.model(input_ids=batch, attention_mask=attn).last_hidden_state[rows, ends])
        s_ids, s_pos, s_qtn, s_last = build_stacked(c_ids, qs)
        ids_t = torch.tensor([s_ids], device=dev)
        pos_t = torch.tensor([s_pos], device=dev)
        last_t = torch.tensor(s_last, device=dev)
        ip = lambda: model.lm_head(model.model(
            input_ids=ids_t, position_ids=pos_t,
            attention_mask=to_additive(ippd_mask(s_pos, s_qtn), model.dtype).to(dev),
        ).last_hidden_state[0, last_t])
        nv = lambda: model.lm_head(model.model(input_ids=ids_t).last_hidden_state[0, last_t])
        t_sep, m_sep = timed(sep)
        t_ip, m_ip = timed(ip)
        t_nv, m_nv = timed(nv)
        row = dict(ctx=len(c_ids), M=M, sep_ms=t_sep, ippd_ms=t_ip, naive_ms=t_nv,
                   sep_mib=m_sep, ippd_mib=m_ip, naive_mib=m_nv,
                   sep_tokens=sum(map(len, seqs)), one_seq_tokens=len(s_ids))
        out["cost"].append(row)
        print(row)

# ---------------------------------------------------------------- toy calibration
def train_logged(loss_fn, steps=cd.STEPS, every=50):
    W = torch.zeros(cd.K, cd.D, requires_grad=True)
    opt = torch.optim.Adam([W], lr=cd.LR)
    g = torch.Generator().manual_seed(1)
    curve = []
    for s in range(steps + 1):
        if s % every == 0:
            curve.append([s, cd.evaluate(W.detach())["kl_to_truth"]])
        if s == steps:
            break
        idx = torch.randint(0, cd.N_TRAIN, (cd.BATCH,), generator=g)
        loss = loss_fn(cd.x_tr[idx] @ W.T, idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return W.detach(), curve


def reliability(W, bins=10):
    with torch.no_grad():
        q = F.softmax(cd.x_te @ W.T, -1)
    conf, pred = q.max(-1)
    correct = (pred == cd.y_te).float()
    edges = torch.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 50:
            rows.append(dict(conf=float(conf[m].mean()), acc=float(correct[m].mean()),
                             n=int(m.sum())))
    return rows


out["calibration"] = {}
for name, fn in [("sl_hard", cd.sl_hard), ("kd_true", cd.kd(1.0)),
                 ("kd_overconf", cd.kd(2.0)), ("rl_gauss", cd.rl_gauss())]:
    W, curve = train_logged(fn)
    out["calibration"][name] = dict(curve=curve, reliability=reliability(W), final=cd.evaluate(W))
    print(name, out["calibration"][name]["final"])

# per-batch cosine between RL and SL gradients, full distribution for a histogram
W0 = (torch.randn(cd.K, cd.D) * 0.3).requires_grad_(True)
cos = []
for b in range(200):
    idx = torch.randint(0, cd.N_TRAIN, (cd.BATCH,))
    g_sl = torch.autograd.grad(cd.sl_hard(cd.x_tr[idx] @ W0.T, idx), W0)[0].flatten()
    g_rl = torch.autograd.grad(cd.rl_gauss()(cd.x_tr[idx] @ W0.T, idx), W0)[0].flatten()
    cos.append(float(F.cosine_similarity(g_sl, g_rl, dim=0)))
out["grad_cosine"] = cos

with open(Path(__file__).resolve().parent / "outputs" / "blog_data.json", "w") as f:
    json.dump(out, f, indent=1)
print("wrote outputs/blog_data.json")
