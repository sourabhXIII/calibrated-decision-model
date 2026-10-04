# calibrated-decision-model

Code for the blog post [Building a fast, calibrated decision model from public parts, and what it can't do](https://sourabhxiii.github.io/blog/2026/calibrated-decision-model/).

Every number and figure in the post comes from these scripts.

## What's here

| file | what it shows | needs |
| --- | --- | --- |
| `ippd_branches.py` | answers several questions about one document in a single forward pass (branch mask plus restarted position IDs), and checks each branch against a separate solo run and against a plain stacked prompt | CUDA GPU |
| `cost_bench.py` | prefill time and peak memory: separate calls vs branches vs plain stacking | CUDA GPU |
| `calib_demo.py` | toy calibration task with known true probabilities: supervised log loss and Brier, distillation from an honest and an overconfident teacher, and Gaussian-noise RL | CPU |
| `blog_data.py` | everything the figures plot, written to `outputs/blog_data.json` | CUDA GPU |
| `figures/make_figures.py` | draws every figure in the post (light and dark SVG) from `outputs/` into `figures/svg/` | standard library only |
| `outputs/` | the outputs of the runs used in the post | |

The language model is [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), downloaded from Hugging Face on first run (about 1 GB).

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python ippd_branches.py --mask-json outputs/toy_mask.json   # exactness check, fp32
python ippd_branches.py --attn sdpa --dtype bf16             # same check in bf16
python cost_bench.py
python calib_demo.py
python blog_data.py                                          # overwrites outputs/blog_data.json
python figures/make_figures.py
```

`ippd_branches.py` exits with an error if the branches don't match the solo runs, so it doubles as a test.

`requirements.txt` pins the versions the post used. Newer versions work too (tested with PyTorch 2.14.1 and Transformers 5.18.0): the fp32 check, the figure data and the calibration results come out the same. The one exception is the bf16 check. Newer PyTorch picks a different kernel for the solo runs, so their label probabilities move by up to 0.03, over the script's 0.02 tolerance, while the branch outputs don't change at all.

## What to expect

The post's numbers came from one NVIDIA GB10 GPU with Python 3.12.3, PyTorch 2.11.0 (CUDA 13.0) and Transformers 5.12.1.

- **fp32:** branch logits match solo runs to about 3e-5. The plain stacked prompt is off by 13.9 and 15.0 logits on Q2 and Q3.
- **bf16:** branches drift from the solo runs by a few tenths of a logit, because tensors of different shapes are summed in a different order. The top answer and the typed label probabilities (within 0.02) still match.
- **Timings** move with hardware and load. The shape should hold: with a 1,241-token context, separate calls grow roughly linearly with the number of questions, while one branched pass barely grows.
- **`calib_demo.py`** is seeded, but exact digits can differ across PyTorch versions and CPUs.

## License

MIT, see [LICENSE](LICENSE).
