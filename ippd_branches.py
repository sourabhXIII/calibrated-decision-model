"""IPPD-style parallel question branches in one forward pass.

One shared context, three questions, one prefill. Each question is a "branch":
it may see the context and its own tokens, never a sibling branch. Position IDs
are virtualised so every branch restarts right after the context, exactly where
it would sit if it had been run alone.

What the script checks
  1. Branch logits from the single stacked pass match three isolated runs.
  2. Naive concatenation (plain causal mask, contiguous positions) does NOT
     match: later questions read earlier ones.
  3. One parallel decode step: append each branch's first answer token and read
     every branch's second-token logits from one more pass.
  4. Typed readout: restrict the first-token distribution to each question's
     label tokens (the way a decision model would).
  5. Token-count cost of N separate calls vs naive concat vs IPPD.

Run (needs a CUDA GPU):
  python ippd_branches.py --mask-json outputs/toy_mask.json   # fp32, eager attention
  python ippd_branches.py --attn sdpa --dtype bf16

Reference: Glavas et al., "Intra-Prompt Parallel Decoding for Common-Context
Question Answering", arXiv:2609.05707 (eqs. 2-10).
"""

from __future__ import annotations

import argparse
import json
import platform

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

CONTEXT = (
    "You read a customer support ticket and answer questions about it. "
    "Answer each question with a single word.\n\n"
    "Ticket: Hi, we were billed twice for March on invoice INV-2207. "
    "Please refund the duplicate charge today, otherwise we will cancel "
    "our plan and move to another provider.\n\n"
)

# Each question ends where the answer would start. `labels` are the typed
# answer options; only their first tokens are used in the typed readout.
QUESTIONS = [
    {"text": "Question: Which department should handle this ticket: billing, technical, or sales?\nAnswer:",
     "labels": [" billing", " technical", " sales"]},
    {"text": "Question: Does the customer threaten to cancel? Answer yes or no.\nAnswer:",
     "labels": [" yes", " no"]},
    {"text": "Question: How urgent is this ticket: low, medium, or high?\nAnswer:",
     "labels": [" low", " medium", " high"]},
]


# --------------------------------------------------------------------------
# Building the stacked sequence, virtual positions and the IPPD mask
# --------------------------------------------------------------------------

def build_stacked(ctx_ids: list[int], q_ids: list[list[int]]):
    """Lay out [context][q1][q2][q3] and tag every token.

    Returns
      input_ids : flat token list
      pos_virt  : virtual position of each token (IPPD eq. 2-5)
      qtn       : question id per token, 0 for the shared context (IPPD eq. 8)
      last_idx  : index of each question's last token (its answer slot)
    """
    input_ids, pos_virt, qtn, last_idx = [], [], [], []
    L = len(ctx_ids)
    input_ids += ctx_ids
    pos_virt += list(range(L))
    qtn += [0] * L
    for k, q in enumerate(q_ids, start=1):
        input_ids += q
        pos_virt += list(range(L, L + len(q)))   # every branch restarts at L
        qtn += [k] * len(q)
        last_idx.append(len(input_ids) - 1)
    return input_ids, pos_virt, qtn, last_idx


def ippd_mask(pos_virt: list[int], qtn: list[int]) -> torch.Tensor:
    """Boolean [L, L] mask, True = query row may attend to key column.

    M_caus(q,k) = pos_virt(q) >= pos_virt(k)
    M_qtn(q,k)  = qtn(q) == qtn(k)  or  qtn(k) == 0
    With a single shared context, IPPD's M_ctx is always true, so it is dropped.
    """
    p = torch.tensor(pos_virt)
    t = torch.tensor(qtn)
    causal = p[:, None] >= p[None, :]
    same_branch_or_shared = (t[:, None] == t[None, :]) | (t[None, :] == 0)
    return causal & same_branch_or_shared


def to_additive(mask_bool: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """[L, L] bool -> [1, 1, L, L] additive mask (0 = keep, min = block)."""
    add = torch.zeros(mask_bool.shape, dtype=dtype)
    add.masked_fill_(~mask_bool, torch.finfo(dtype).min)
    return add[None, None]


@torch.no_grad()
def forward_logits(model, ids, positions=None, mask_bool=None):
    """Logits for every position of one sequence."""
    dev = model.device
    input_ids = torch.tensor([ids], device=dev)
    kwargs = {}
    if positions is not None:
        kwargs["position_ids"] = torch.tensor([positions], device=dev)
    if mask_bool is not None:
        kwargs["attention_mask"] = to_additive(mask_bool, model.dtype).to(dev)
    return model(input_ids=input_ids, **kwargs).logits[0].float()


# --------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--mask-json", default=None,
                    help="optional path to dump a small toy mask for the diagram")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible; this script needs one.")
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=dtype, attn_implementation=args.attn).to("cuda").eval()

    print(f"python {platform.python_version()} | torch {torch.__version__} | "
          f"transformers {transformers.__version__} | {torch.cuda.get_device_name(0)}")
    print(f"model {MODEL} | dtype {args.dtype} | attention {args.attn}\n")

    ctx_ids = tok(CONTEXT, add_special_tokens=False)["input_ids"]
    q_ids = [tok(q["text"], add_special_tokens=False)["input_ids"] for q in QUESTIONS]
    # Sanity: tokenising context and question separately must equal tokenising
    # the joined string, otherwise "isolated" and "stacked" see different tokens.
    for q, qi in zip(QUESTIONS, q_ids):
        joined = tok(CONTEXT + q["text"], add_special_tokens=False)["input_ids"]
        assert joined == ctx_ids + qi, "token boundary merges across context/question"

    ids, pos_virt, qtn, last_idx = build_stacked(ctx_ids, q_ids)
    mask = ippd_mask(pos_virt, qtn)
    print(f"context {len(ctx_ids)} tok | questions {[len(q) for q in q_ids]} tok | "
          f"stacked length {len(ids)}")

    # 1) IPPD: one pass over the stacked sequence
    ippd = forward_logits(model, ids, pos_virt, mask)

    # 2) Isolated: one ordinary causal pass per question
    iso = [forward_logits(model, ctx_ids + q)[-1] for q in q_ids]

    # 3) Naive concatenation: plain causal mask, contiguous positions
    naive = forward_logits(model, ids)

    print("\n[1] First answer token: IPPD branch vs isolated run")
    # fp32: logits must agree to 1e-3. bf16: the stacked and isolated runs have
    # different shapes, so kernels reduce in a different order and logits drift
    # by tenths (IPPD paper, App. C.1). There we require the same argmax and
    # typed label probabilities within 0.02.
    tol = 1e-3
    results = []
    for k, (li, iso_l) in enumerate(zip(last_idx, iso), start=1):
        d_ippd = (ippd[li] - iso_l).abs().max().item()
        d_naive = (naive[li] - iso_l).abs().max().item()
        same_top_ippd = ippd[li].argmax().item() == iso_l.argmax().item()
        same_top_naive = naive[li].argmax().item() == iso_l.argmax().item()
        top = tok.decode([iso_l.argmax().item()])
        results.append(dict(q=k, max_abs_diff_ippd=d_ippd, max_abs_diff_naive=d_naive,
                            same_argmax=same_top_ippd))
        print(f"  Q{k}: isolated top token {top!r:12} | max|Δlogit| IPPD {d_ippd:.2e} "
              f"(argmax same: {same_top_ippd}) | naive concat {d_naive:.2e} "
              f"(argmax same: {same_top_naive})")
    if args.dtype == "fp32":
        ok = all(r["max_abs_diff_ippd"] < tol for r in results)
        print(f"  -> IPPD matches isolated runs within {tol:g}: {ok}")
    else:
        ok = all(r["same_argmax"] for r in results)
        print(f"  -> bf16: same argmax on every branch: {ok} (typed probabilities checked below)")
    print("  (Q1 is unaffected by naive concat because nothing comes before it; "
          "Q2 and Q3 read their predecessors.)")

    # 4) Typed readout: softmax restricted to each question's label tokens
    print("\n[2] Typed readout (softmax over each question's label first-tokens)")
    for k, (q, li, iso_l) in enumerate(zip(QUESTIONS, last_idx, iso), start=1):
        lab_ids = [tok(l, add_special_tokens=False)["input_ids"][0] for l in q["labels"]]
        p_ippd = torch.softmax(ippd[li, lab_ids], -1)
        p_iso = torch.softmax(iso_l[lab_ids], -1)
        p_naive = torch.softmax(naive[li, lab_ids], -1)
        if (p_ippd - p_iso).abs().max().item() > 0.02:
            ok = False
        mass = torch.softmax(iso_l, -1)[lab_ids].sum().item()
        fmt = lambda p: " ".join(f"{l.strip()}={v:.3f}" for l, v in zip(q["labels"], p.tolist()))
        print(f"  Q{k}: IPPD    {fmt(p_ippd)}")
        print(f"      isolated {fmt(p_iso)}")
        print(f"      naive    {fmt(p_naive)}")
        print(f"      (label tokens hold {mass:.1%} of the full-vocabulary mass before renormalising)")
        # Renormalisation bias in miniature (arXiv:2605.09739): the model may
        # put its mass on " No" or " Medium" rather than the exact label token.
        # Pooling simple case/space variants recovers part of that "silent vote".
        full = torch.softmax(iso_l, -1)
        pooled = []
        for l in q["labels"]:
            w = l.strip()
            variants = {f" {w}", f" {w.capitalize()}", w, w.capitalize()}
            vids = {tok(v, add_special_tokens=False)["input_ids"][0] for v in variants}
            pooled.append(full[list(vids)].sum())
        pooled = torch.stack(pooled)
        print(f"      variant-pooled {fmt(pooled / pooled.sum())} "
              f"(covers {pooled.sum().item():.1%} of the mass)")

    # 5) One parallel decode step. Append each branch's greedy first token,
    #    tagged with its branch id and the next virtual position, then read
    #    every branch's second-token logits from a single pass.
    print("\n[3] Parallel decode step (second answer token)")
    first = [ippd[li].argmax().item() for li in last_idx]
    ids2, pos2, qtn2 = list(ids), list(pos_virt), list(qtn)
    new_idx = []
    for k, (t, li) in enumerate(zip(first, last_idx), start=1):
        ids2.append(t)
        pos2.append(pos_virt[li] + 1)
        qtn2.append(k)
        new_idx.append(len(ids2) - 1)
    ippd2 = forward_logits(model, ids2, pos2, ippd_mask(pos2, qtn2))
    for k, (q, t, ni) in enumerate(zip(q_ids, first, new_idx), start=1):
        iso2 = forward_logits(model, ctx_ids + q + [t])[-1]
        d = (ippd2[ni] - iso2).abs().max().item()
        print(f"  Q{k}: after {tok.decode([t])!r:12} max|Δlogit| vs isolated {d:.2e}")

    # 6) Cost in tokens pushed through the model for the prefill
    Lc, Lq = len(ctx_ids), [len(q) for q in q_ids]
    sep = sum(Lc + l for l in Lq)
    stacked = Lc + sum(Lq)
    print("\n[4] Prefill tokens processed")
    print(f"  N separate calls : {sep}")
    print(f"  naive concat     : {stacked} (cheap, but leaks across questions)")
    print(f"  IPPD             : {stacked} (same tokens as naive, no leakage)")
    print(f"  ratio separate / IPPD = {sep / stacked:.2f}x")

    if args.mask_json:
        # A small toy layout for the diagram: context 6, questions 3, 2, 4.
        t_ids, t_pos, t_qtn, _ = build_stacked(list(range(6)), [[0] * 3, [0] * 2, [0] * 4])
        t_mask = ippd_mask(t_pos, t_qtn).int().tolist()
        with open(args.mask_json, "w") as f:
            json.dump({"pos_virt": t_pos, "qtn": t_qtn, "mask": t_mask}, f)
        print(f"\nwrote toy mask to {args.mask_json}")

    if not ok:
        raise SystemExit("IPPD branches did not match isolated runs")


if __name__ == "__main__":
    main()
