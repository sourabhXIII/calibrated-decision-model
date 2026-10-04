"""Prefill cost of answering M questions about one shared context.

Three ways to get the first answer token of M questions:
  separate : M sequences [context + q_k] in one padded batch (no sharing)
  ippd     : one stacked sequence [context][q_1]...[q_M] with the IPPD mask
  naive    : the same stacked sequence with a plain causal mask (leaks; shown
             only as the speed ceiling of "one sequence, no masking work")

Measures wall-clock prefill time and peak memory on one GPU. Single-token
answers make prefill the whole job, which is the regime a typed-decision model
lives in. Prefix caching (vLLM) is not reproduced here; see IPPD Sec. 6 for
that comparison.

Run (needs a CUDA GPU):  python cost_bench.py
"""

from __future__ import annotations

import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from ippd_branches import MODEL, build_stacked, ippd_mask, to_additive

TICKET = ("Hi, we were billed twice for March on invoice INV-2207. Please refund the "
          "duplicate charge today, otherwise we will cancel our plan. ")
QUESTION = "Question: Is fact number {i} in the ticket above true? Answer yes or no.\nAnswer:"


@torch.no_grad()
def timed(fn, reps=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3, torch.cuda.max_memory_allocated() / 2**20


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    dev = model.device
    print(f"torch {torch.__version__} | transformers {transformers.__version__} | "
          f"{torch.cuda.get_device_name(0)} | bf16 sdpa\n")
    print(f"{'ctx':>5} {'M':>3} | {'tokens sep':>10} {'tokens 1-seq':>12} | "
          f"{'separate ms':>11} {'ippd ms':>8} {'naive ms':>8} | {'sep/ippd':>8} | "
          f"{'sep MiB':>8} {'ippd MiB':>8}")

    for ctx_reps in (4, 40):                       # 125 and 1,241 context tokens
        ctx_ids = tok(TICKET * ctx_reps + "\n\n", add_special_tokens=False)["input_ids"]
        for M in (1, 4, 16, 32):
            q_ids = [tok(QUESTION.format(i=i), add_special_tokens=False)["input_ids"]
                     for i in range(M)]

            # separate: right-padded batch, read logits at each row's last real token
            seqs = [ctx_ids + q for q in q_ids]
            Lmax = max(map(len, seqs))
            batch = torch.full((M, Lmax), tok.pad_token_id, device=dev)
            attn = torch.zeros((M, Lmax), dtype=torch.long, device=dev)
            for r, s in enumerate(seqs):
                batch[r, :len(s)] = torch.tensor(s)
                attn[r, :len(s)] = 1
            rows = torch.arange(M, device=dev)
            ends = torch.tensor([len(s) - 1 for s in seqs], device=dev)
            # run the LM head only at the answer slots, for every method alike
            sep = lambda: model.lm_head(
                model.model(input_ids=batch, attention_mask=attn).last_hidden_state[rows, ends])

            ids, pos, qtn, last = build_stacked(ctx_ids, q_ids)
            ids_t = torch.tensor([ids], device=dev)
            pos_t = torch.tensor([pos], device=dev)
            # the mask is rebuilt inside the timed call, so its cost is counted
            last_t = torch.tensor(last, device=dev)
            ippd = lambda: model.lm_head(model.model(
                input_ids=ids_t, position_ids=pos_t,
                attention_mask=to_additive(ippd_mask(pos, qtn), model.dtype).to(dev),
            ).last_hidden_state[0, last_t])
            naive = lambda: model.lm_head(model.model(input_ids=ids_t).last_hidden_state[0, last_t])

            t_sep, m_sep = timed(sep)
            t_ippd, m_ippd = timed(ippd)
            t_naive, _ = timed(naive)
            n_sep, n_one = sum(map(len, seqs)), len(ids)
            print(f"{len(ctx_ids):>5} {M:>3} | {n_sep:>10} {n_one:>12} | "
                  f"{t_sep:>11.1f} {t_ippd:>8.1f} {t_naive:>8.1f} | {t_sep / t_ippd:>7.2f}x | "
                  f"{m_sep:>8.0f} {m_ippd:>8.0f}")


if __name__ == "__main__":
    main()
