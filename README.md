# 🖼️ Readymades

**The Gallery**
Est. 2025 · Every 15 Minutes

---

## 🎨 The Collection

Duchamp put a urinal in a gallery. We put datasets.

Same data. Wrong order. New meaning. No meaning.

Curator #0 never stops. Every 15 minutes. New exhibition. New edition.

---

## 🏛️ Current Exhibition

| Edition | Original | Medium | Exhibited |
|---------|----------|--------|-----------|
| #1 | [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) | 10 cols · 500 rows | 2025-11-02 |
---

## 🎭 The Method

```
Select random dataset → Check constraints → Stream data → Shuffle columns → Upload → Archive
```

**Selection Strategy:**
- 40% Popular datasets (1000+ downloads)
- 30% Medium reach (100-1000 downloads)
- 20% Emerging (10-100 downloads)
- 10% Random finds

**Constraints (GitHub Actions limits):**
- Max 50 MB dataset size
- Max 100 files (skip image/audio datasets)
- Stream only 500 rows
- 60 second timeout

**Process:**
- Each column shuffled independently
- All row relationships destroyed
- Structure preserved (keep types)
- Cache cleaned between runs
- Re-upload as readymade
- Original credited

---

## 📦 Request Exhibition

Want a specific dataset curated?

📧 **hi@sdpkjc.com**

---

## 🔧 Run Your Own

```bash
pip install -r requirements.txt
export HF_TOKEN="your-token"
python curator0.py  # Start Curator #0
```

Or let GitHub Action run it automatically.

---

> _"I am a machine."_
> — Andy Warhol
>
> _"We remix machines."_
> — TheFactoryX

**[TheFactoryX](https://github.com/TheFactoryX)** — Strange people. Strange things.


## 🖼️ Gallery Archive

| Edition # | Timestamp | Original | Process | Readymade |
|-----------|-----------|----------|---------|-----------|
| 0 | 2025-11-02 13:42:23 | [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news) | streamed (500 rows, 2 cols) | [edition_0000_fancyzhx-ag_news](edition_0000_fancyzhx-ag_news) |
| 1 | 2025-11-02 13:47:47 | [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) | streamed (500 rows, 10 cols) | [edition_0001_Rowan-hellaswag](edition_0001_Rowan-hellaswag) |
