<div align="center">

#  Insurance Taxonomy Classifier  
### High-Precision NLP-Based Company-to-Insurance Label Mapping

Lightweight. Interpretable. Scalable. Production-ready.

</div>

---

## Overview

**Insurance Taxonomy Classifier** is a high-performance NLP-based system designed to map companies to structured insurance taxonomy labels using hybrid TF-IDF vectorization and cosine similarity.

It is specifically built for:

- Large company datasets  
- Noisy or inconsistent textual descriptions  
- Closely related taxonomy labels  
- Fast and memory-efficient processing (sparse matrices)  
- Fully explainable classification logic  

No deep learning. No black boxes. Fully auditable.

---

## How It Works

### 1️. Text Aggregation  
Each company is represented using:

- `description`
- `sector`
- `category`
- `niche`
- `business_tags`

All fields are cleaned, normalized, and merged into a single structured text representation.

---

### 2️. Hybrid Vectorization

The model combines two complementary TF-IDF representations:

| Type | Purpose |
|------|---------|
| Word n-grams (1–2) | Captures semantic meaning |
| Character n-grams (3–5) | Captures spelling variations & robustness |

The vectors are:

- Horizontally stacked (`scipy.sparse`)
- L2-normalized
- Compared using cosine similarity

---

### 3️. Smart Label Centroids (Seed-Based)

Instead of relying only on raw label text:

- The model detects **seed companies** strongly matching a label
- If enough seeds exist → builds a **centroid representation**
- Otherwise → falls back to label text vector

This significantly improves classification quality.

---

### 4️. Multi-Label Logic

Additional labels are assigned when:

- Similarity ≥ `add_label_min_score`
- Close enough to best label (`add_label_close_ratio`)

This enables intelligent multi-label classification.

---

## Installation

```bash
pip install numpy pandas scipy scikit-learn
