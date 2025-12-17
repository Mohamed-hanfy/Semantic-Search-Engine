# Vector Voyage — Semantic Search Engine

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EG9mSJFLtUzis7wp9RNfI21batf_36Ca?usp=sharing)

A fast, semantic search engine that preprocesses text, builds vector embeddings using transformer models, and performs efficient nearest-neighbor searches. Built on the Quora Questions Dataset with 404K+ question pairs.

##  Features

- **Efficient Embedding Generation**: Uses the lightweight `all-MiniLM-L6-v2` transformer model for fast, high-quality embeddings
- **Scalable Search**: KNN-based search with pre-computed neighbor indices
- **Deduplication**: Automatically filters duplicate questions from the dataset
- **Fast Queries**: Average query latency under 50ms on CPU
- **Batch Processing**: Support for batch queries with linear throughput scaling
- **Comprehensive Benchmarking**: Built-in performance metrics and visualizations
- **GPU/CPU Support**: Automatic device detection and optimization

##  Dataset

**Quora Questions Dataset**
- **Size**: 404,302 question pairs
- **Format**: CSV with columns: `id`, `qid1`, `qid2`, `question1`, `question2`, `is_duplicate`
- **Coverage**: Real-world question pairs with semantic similarity labels
- **Preprocessing**: Automatic deduplication and unique question extraction

Sample data:
```
id, qid1, qid2, question1, question2, is_duplicate
0,  1,    2,    "What is the step by step guide to invest in share market in india?", 
              "What is the step by step guide to invest in share market?", 0
```

## Architecture

### 1. Data Preparation
- Loads the Quora dataset from CSV
- Filters duplicate question pairs (`is_duplicate == 0`)
- Extracts unique questions and builds a text index
- Deduplicates question texts while preserving order

### 2. Embedding Generation
- Model: `sentence-transformers/all-MiniLM-L6-v2` (12 layers, 384 dimensions)
- Batch size: 128 for GPU optimization
- Caching: Pre-computed embeddings saved to `quora_embeddings.npy`
- Normalization: L2 normalized embeddings for consistent similarity scores

### 3. Indexing & Search
- Algorithm: K-Nearest Neighbors with Euclidean metric
- Pre-computation: Indexes up to 50 neighbors per query point
- CPU Parallelization: Uses all available CPU cores
- Metric: Euclidean distance (lower = more similar)

### 4. Query Processing
- Query encoding: Real-time embedding generation
- Retrieval: KNN search in pre-computed index
- Results: Ranked by distance (similarity score)

## Performance Benchmarks

### Query Performance
| Metric | Value |
|--------|-------|
| Average Query Latency | ~30-40 ms |
| Median Query Latency | ~25-35 ms |
| Encoding Time (GPU) | ~15-20 ms |
| KNN Search Time (CPU) | ~10-20 ms |
| Queries Per Second | 25-30 QPS |

### Throughput
- **Single Query**: 30-40 ms
- **Batch (10 queries)**: ~3 ms per query

### Memory Usage
- **Embeddings**: ~1.5 GB for 404K questions (384-dim vectors)
- **Per Question**: ~4 KB
- **KNN Index**: ~1.8 GB (estimated)
- **Total**: ~3.3 GB

