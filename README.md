<div align="center">

# Text Generation: Statistical vs. Neural vs. Transformer
### Comparing n-gram, RNN/LSTM, and Fine-Tuned GPT-2 on the Shakespeare Corpus

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/-NLP-8A2BE2?style=flat)
![Transformers](https://img.shields.io/badge/-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)

*Three generations of language modeling, one dataset: does more capacity actually mean better text?*

</div>

---

## Overview

Three very different ideas about how to predict "what comes next," trained and evaluated on the **same** Shakespeare corpus so the comparison is actually fair:

- 1) **Statistical n-gram** : frequency counts + backoff smoothing
- 2) **Character-level RNN & LSTM** : trained from scratch
- 3) **Fine-tuned GPT-2** : pretrained transformer adapted to Shakespeare

Each is scored on **fluency** (perplexity) and **diversity** (Self-BLEU) to see where each paradigm actually wins — and where bigger isn't better.

> 📄 Full write-up: [`Project-Report.pdf`](assets/Text-Gen-report.pdf) — literature review, methodology, references
> 👥 Group project — with **Aqsa Mohsin** - for *Knowledge Processing in Intelligent Systems* (WiSe 2025/26), Knowledge Technology (WTM), University of Hamburg

---

## Models

<details>
<summary><strong>Statistical n-gram (backoff, add-k smoothing)</strong></summary>

```
Sentence → tokenize → pad with (n−1) <s> and one </s>
For each order k = 1..n: count (context, word) pairs → C(c, w), C(c)

P(w | c) = ( C(c, w) + α ) / ( C(c) + α·|V| )        [add-k smoothing]

Inference: try the highest order with a non-zero count,
           back off to lower orders until unigram if unseen
Generation: greedy or temperature sampling from the backoff
            distribution, until </s> or max length
```
Word-level, orders n ∈ {1, 2, 3}. Rare tokens (below a min-count threshold) map to `<unk>`.
</details>

<details>
<summary><strong>Character-level RNN & LSTM (trained from scratch)</strong></summary>

```
Character index → Embedding (dim 128)
  → Recurrent layer, 256 hidden units (SimpleRNN or LSTM)
  → Fully connected + softmax over 65-character vocabulary
  → next-character prediction, generated autoregressively
```
Same architecture for both — the only difference is the recurrent cell, isolating exactly what LSTM's gating buys you over a vanilla RNN.
</details>

<details>
<summary><strong>Fine-tuned GPT-2</strong></summary>

```
Shakespeare corpus → GPT-2 BPE tokenizer → continuous token stream
  → split into 128-token blocks, causal attention masks
  → fine-tune pretrained GPT-2 (causal language modeling objective)
  → generate autoregressively, up to 200 tokens
```
</details>

---

## Dataset

The **Shakespeare corpus** (TensorFlow), ~1.1M characters, 65 unique symbols. Minimal preprocessing to preserve original structure and style. **Same 80% / 10% / 10%** train / validation / eval split used across all three model families for a fair comparison.

## Results

> ⚠️ Each model is evaluated at its **own** token granularity — n-gram is word-level, RNN/LSTM are character-level, GPT-2 is subword-level — so perplexity numbers are only directly comparable *within* the same granularity, not across the table.

**n-gram (word-level):**

| Model | PPL | Fallback rate | Vocab |
|---|:---:|:---:|:---:|
| Unigram (n=1) | 264.10 | 1.000 | 5,796 |
| **Bigram (n=2)** | **146.20** | 0.190 | 5,796 |
| Trigram (n=3) | 238.91 | 0.190 | 5,796 |

**Neural & transformer:**

| Model | PPL | Eval set | Token level |
|---|:---:|---|---|
| RNN | 6.63 | Shakespeare | Character |
| **LSTM** | **2.67** | Shakespeare | Character |
| GPT-2 | 31.35 | Shakespeare | Subword |
| GPT-2 | 194.45 | Wikipedia (out-of-domain) | Subword |

**Self-BLEU (diversity, lower = more diverse):** n-gram is the least diverse (67.76 on bigram), GPT-2 next, then RNN, with **LSTM the most diverse of all models tested**.

**Key takeaways:**
- Bigrams cut perplexity hugely over unigrams but trigrams get *worse*, not better: with this corpus size, higher-order contexts are too sparse for the backoff smoothing to fully compensate.
- **LSTM wins on both axes** : lowest perplexity *and* most diverse output — the best fluency/diversity trade-off of any model tested.
- Fine-tuned **GPT-2 generalizes poorly out-of-domain**: perplexity jumps from 31.35 -> 194.45 (~6×) when evaluated on Wikipedia-style text instead of Shakespeare, despite being the largest, most pretrained model in the comparison.
- **Bigger model ≠ better fit**: on a small, stylistically narrow corpus, a from-scratch LSTM beat a pretrained transformer on both fluency and diversity - capacity matters less than architecture/domain alignment here.


---

## References
[1] Peter F. Brown, Vincent J. Della Pietra, Peter V. deSouza, Jenifer C. Lai, and Robert L. Mercer. Class-based n-gram models of natural language. Computational Linguistics, 18(4):467–480, 1992.

[2] Sepp Hochreiter and Jurgen Schmidhuber. Long short-term memory. ¨
Neural Comput., 9(8):1735–1780, November 1997.

[3] Andrej Karpathy. The unreasonable effectiveness of recurrent neural
networks, 2015.


*(full reference list in the report)*

## 👥 Authors
**Laiba Qureshi** & **Aqsa Mohsin** — Knowledge Processing in Intelligent Systems, University of Hamburg
