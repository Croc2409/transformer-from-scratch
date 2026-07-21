# Transformer From Scratch

Transformer built from scratch in PyTorch — standard attention (Vaswani et al., 2017) + relative position variant (Shaw et al., 2018), trained on sequence reversal. No `nn.Transformer` or `nn.MultiheadAttention` used — every component (multi-head attention, masking, positional encoding, encoder/decoder stack, autoregressive decoding) is implemented from scratch.

## What's Inside

This repo contains two implementations trained on the same task (sequence reversal), letting you directly compare standard absolute-position attention against relative position attention:

- **`standard-attention/`** — Classic scaled dot-product multi-head attention with sinusoidal absolute positional encodings, as described in *Attention Is All You Need* (Vaswani et al., 2017).
- **`relative-position/`** — Relative Position Representations (RPR), as described in *Self-Attention with Relative Position Representations* (Shaw et al., 2018). Instead of injecting position via input embeddings, relative position information is added directly into the attention score and value computations, letting the model reason about *how far apart* tokens are rather than their absolute index.

Both include custom causal masking and greedy autoregressive decoding at inference time, built without relying on high-level PyTorch transformer modules.

## Results

| Model | Accuracy |
|---|---|
| Standard attention (Vaswani et al., 2017) | 99.84% |
| Relative position attention (Shaw et al., 2018) | 99.93% (in-distribution) |

Both models were trained and evaluated on a sequence reversal task, where the model must learn to output a reversed copy of an input token sequence — a good stress test for positional understanding, since it requires precise per-position reasoning rather than pattern-matching on content alone.

## Architecture Details

- Multi-head self-attention implemented from first principles (query/key/value projections, scaled dot-product, softmax, head concatenation)
- Custom padding and causal (look-ahead) masking for autoregressive decoding
- Encoder-decoder stack built manually, no `nn.Transformer`
- RPR variant adds learned relative position embeddings directly into attention logits and value aggregation, per Shaw et al. (2018)
- Greedy autoregressive decoding loop for inference

## Setup & Usage

```bash
git clone https://github.com/<your-username>/transformer-from-scratch.git
cd transformer-from-scratch

pip install -r requirements.txt

# Train standard attention variant
python standard-attention/train.py

# Train relative position variant
python relative-position/train.py
```


## References

- Vaswani, A. et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)

## License

MIT



