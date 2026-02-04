
# Word2Vec Notes — CBOW & Skip-gram (with full math + examples)

> **Big picture:** Word2Vec is a tiny neural net trained on a self-supervised task.
> - **CBOW:** predict the **center/target** word from its **context**.
> - **Skip-gram:** predict the **context** words from the **center/target** word.
>
> In both cases, the *learned parameters* are word embedding matrices. After training, you keep the embeddings.

---

## 0) Notation / Setup

- Corpus is a sequence of tokens:  
  \[
  w_1, w_2, \dots, w_T
  \]
- Vocabulary size: \(V\). Embedding dimension: \(d\).
- One-hot vector for word \(w\): \(x_w \in \mathbb{R}^{V}\) (all zeros except a 1 at index \(w\)).
- Window radius: \(c\).  
  For position \(t\), the context indices are:
  \[
  \{t-c, \dots, t-1, t+1, \dots, t+c\} \cap [1,T]
  \]
- Context multiset at position \(t\):
  \[
  C_t = \{ w_{t+j} : j \in [-c,c],\ j\neq 0,\ 1\le t+j \le T\}
  \]

---

## 1) Parameters (same for CBOW & Skip-gram)

Word2Vec maintains **two** embedding tables:

1) **Input embedding matrix**
\[
W \in \mathbb{R}^{V \times d}
\]
Row \(W[w]\) is the embedding used when word \(w\) appears on the **input** side.

2) **Output embedding matrix**
\[
U \in \mathbb{R}^{d \times V}
\]
Column \(U[:,w]\) is the embedding used when word \(w\) appears on the **output** side.

> Many implementations store \(U^\top \in \mathbb{R}^{V\times d}\) instead. Same thing, just transposed.

After training, common choices for final embeddings:
- Use \(W\) (input vectors) **most common**
- Or use \(U^\top\) (output vectors)
- Or combine: \(E = W + U^\top\) or average \((W + U^\top)/2\)

---

## 2) The shared scoring model (logits)

Given a hidden/context representation \(h \in \mathbb{R}^{d}\), produce logits over all vocab words:

\[
z = hU \in \mathbb{R}^{V}
\]
and for a candidate word \(w\):
\[
z_w = h^\top U[:,w]
\]

Convert logits to probabilities via softmax:

\[
p(w \mid h) = \frac{e^{z_w}}{\sum_{j=1}^{V} e^{z_j}}
\]

---

## 3) CBOW (Continuous Bag of Words)

### 3.1 Training example construction
At position \(t\):
- **Input:** context words \(C_t\)
- **Target:** center word \(w_t\)

### 3.2 Forward pass (full softmax version)

**Step A — embed each context word**
For each context word \(w \in C_t\), its input embedding is \(W[w]\).

**Step B — aggregate context vectors**
CBOW uses *bag-of-words* (orderless). Typical aggregation is mean:

\[
h = \frac{1}{|C_t|}\sum_{w\in C_t} W[w] \in \mathbb{R}^{d}
\]

(Using sum instead of mean changes scaling but not the core idea.)

**Step C — compute logits and probability of the target**
\[
z = hU
\]
\[
p(w_t \mid C_t) = \frac{e^{h^\top U[:,w_t]}}{\sum_{j=1}^{V} e^{h^\top U[:,j]}}
\]

### 3.3 Loss (cross-entropy)
For one training position \(t\):

\[
\mathcal{L}_{\text{CBOW}}(t) = -\log p(w_t \mid C_t)
\]

Overall objective over corpus:
\[
\min \frac{1}{T}\sum_{t=1}^T \mathcal{L}_{\text{CBOW}}(t)
\]

### 3.4 What updates in CBOW?
- **All** context input rows \(W[w]\) for \(w\in C_t\) get updated (because they contributed to \(h\)).
- Many output columns in \(U\) are involved if using full softmax (in practice: approximations).

**Intuition:** make \(h\) align with \(U[:,w_t]\) and misalign with other output vectors.

---

## 4) Skip-gram

### 4.1 Training example construction (skip-grams)
At position \(t\):
- **Input / target:** the center word \(w_t\)
- **Outputs:** each context word in \(C_t\)

Instead of one example per position, Skip-gram produces **pairs**:
\[
(w_t, w_{t+j}) \quad \text{for each valid } j\neq 0,\ |j|\le c
\]

These pairs are called **skip-grams**.

### 4.2 Forward pass (full softmax version) for one pair \((t, c)\)
Let target word be \(t\) and one context word be \(c\).

**Step A — lookup target embedding**
Because input is one-hot, multiplying by \(W\) selects a row:

\[
h = W[t] \in \mathbb{R}^{d}
\]

**Step B — logits and probability of the context**
\[
z = hU
\]
\[
p(c \mid t) = \frac{e^{h^\top U[:,c]}}{\sum_{j=1}^{V} e^{h^\top U[:,j]}}
\]

### 4.3 Loss (cross-entropy)
For one pair \((t,c)\):

\[
\mathcal{L}_{\text{SG}}(t,c) = -\log p(c \mid t)
\]

For a center position \(t\) with multiple context words:
\[
\mathcal{L}_{\text{SG}}(t) = \sum_{c \in C_t} -\log p(c \mid w_t)
\]

### 4.4 What updates in Skip-gram?
For each pair \((t,c)\):
- Only **one input row** \(W[t]\) updates (the center word).
- Output matrix \(U\) updates (full softmax touches many columns; negative sampling touches few).

**Intuition:** make \(W[t]\) align with \(U[:,c]\) for true contexts, and not align with others.

---

## 5) Key difference: CBOW vs Skip-gram (mechanically)

### CBOW
- One training instance per position (predict center from the set of context words).
- Input representation: average of multiple embeddings.
- Updates: many \(W[w]\) rows (all context words).

### Skip-gram
- Many training instances per position (one per context word).
- Input representation: a single embedding \(W[t]\).
- Updates: one \(W[t]\) row per pair.

---

## 6) Example (the classic sentence)

Sentence:
> "the wide road shimmered in the hot sun"

Tokens:
0 the, 1 wide, 2 road, 3 shimmered, 4 in, 5 the, 6 hot, 7 sun

Window radius \(c=2\).

### 6.1 Skip-gram pairs (partial)
For target index 2 = "road", context indices are {0,1,3,4}:
- (road, the)
- (road, wide)
- (road, shimmered)
- (road, in)

For target index 3 = "shimmered", context {1,2,4,5}:
- (shimmered, wide)
- (shimmered, road)
- (shimmered, in)
- (shimmered, the)

Total pairs count:
Edges have fewer contexts. Summing counts gives 26 pairs for this sentence with \(c=2\).

### 6.2 CBOW examples (partial)
For center word "road" (index 2), CBOW input context is:
- {the, wide, shimmered, in}  → predict "road"

For center word "shimmered" (index 3):
- {wide, road, in, the} → predict "shimmered"

---

## 7) Why full softmax is expensive

Computing:
\[
p(w \mid h) = \frac{e^{h^\top U[:,w]}}{\sum_{j=1}^{V} e^{h^\top U[:,j]}}
\]
requires summing over **all \(V\)** words.

Time per example is \(O(V)\), too expensive for \(V \sim 10^5 - 10^7\).

So we use approximations:
- **Negative Sampling** (most common in word2vec)
- **Hierarchical Softmax** (tree-based)

---

## 8) Negative Sampling (core practical Word2Vec)

Instead of multiclass softmax, train a **binary classifier**:
- Positive pair: (target, true_context) should be labeled 1
- Negative pairs: (target, noise_word) labeled 0

Let:
- target word = \(t\)
- positive context = \(c\)
- negatives = \(n_1, \dots, n_K\) sampled from a noise distribution \(P_n\)

Define sigmoid:
\[
\sigma(x) = \frac{1}{1+e^{-x}}
\]

### 8.1 Skip-gram with Negative Sampling (SGNS)

Score for pair \((t,w)\):
\[
s(t,w) = W[t] \cdot U[:,w]
\]

Loss for one positive + K negatives:
\[
\mathcal{L}_{\text{SGNS}}(t,c) =
-\log \sigma(s(t,c))
-\sum_{i=1}^{K} \log \sigma(-s(t,n_i))
\]

**Interpretation:**
- Make \(W[t]\cdot U[:,c]\) large (positive)
- Make \(W[t]\cdot U[:,n_i]\) small/negative

**What gets updated per step:**
- \(W[t]\) (one row)
- \(U[:,c]\) and \(U[:,n_i]\) (only K+1 columns)

This is why it’s fast.

### 8.2 CBOW with Negative Sampling
CBOW hidden vector:
\[
h = \frac{1}{|C_t|}\sum_{w\in C_t} W[w]
\]
Score:
\[
s(h,w) = h \cdot U[:,w]
\]
Loss:
\[
\mathcal{L}_{\text{CBOW-NS}}(C_t, w_t) =
-\log \sigma(h\cdot U[:,w_t])
-\sum_{i=1}^{K}\log \sigma(-h\cdot U[:,n_i])
\]

Updates:
- all \(W[w]\) for \(w\in C_t\)
- output vectors for \(w_t\) and negatives

---

## 9) (Optional but useful) Gradients: what “moves where”

### 9.1 Softmax + cross-entropy (Skip-gram, one pair)
Let \(p_j = p(j \mid t)\). True label is one-hot \(y_c=1\).

Gradient wrt output columns:
\[
\frac{\partial \mathcal{L}}{\partial U[:,j]} = W[t]\,(p_j - y_j)
\]

- For true context \(c\): \(p_c-1 < 0\) ⇒ move \(U[:,c]\) **toward** \(W[t]\)
- For others: \(p_j > 0\) ⇒ move \(U[:,j]\) slightly **away** from \(W[t]\)

Gradient wrt input row:
\[
\frac{\partial \mathcal{L}}{\partial W[t]} = \sum_{j=1}^{V} U[:,j](p_j - y_j)
\]

### 9.2 Negative sampling (Skip-gram, one positive + K negatives)
Let \(u_w = U[:,w]\), \(v_t = W[t]\).

For the positive pair:
- push \(v_t\) and \(u_c\) together

For each negative:
- push \(v_t\) and \(u_{n_i}\) apart

This is exactly “attract positive, repel negatives”.

---

## 10) Training algorithms (high-level pseudocode)

### 10.1 Skip-gram (with negative sampling)


for each position t in corpus:
target = w_t
for each context c in C_t:
sample negatives n1..nK ~ Pn
update W[target], U[c], U[n1..nK] to minimize:
-log σ(W[target]·U[c]) - Σ log σ(-W[target]·U[ni])



### 10.2 CBOW (with negative sampling)


for each position t in corpus:
context_words = C_t
h = average(W[w] for w in context_words)
sample negatives n1..nK ~ Pn
update W[context_words], U[target], U[n1..nK] to minimize:
-log σ(h·U[target]) - Σ log σ(-h·U[ni])



---

## 11) Practical details often used with Word2Vec

### 11.1 Subsampling frequent words
Very frequent words (the, of, and) dominate pairs.
A common trick drops word \(w\) with probability:
\[
P(\text{drop } w) = 1 - \sqrt{\frac{t}{f(w)}}
\]
where \(f(w)\) is frequency and \(t\) is a small threshold (e.g. \(10^{-5}\)).

### 11.2 Negative sampling distribution
Negatives are sampled proportional to:
\[
P_n(w) \propto f(w)^{3/4}
\]
(Heuristic that works well in practice.)

### 11.3 Dynamic window
Instead of fixed radius \(c\), sample an effective window size uniformly from \([1,c]\) per center word.

---

## 12) Summary cheat sheet

### CBOW
- Input: many context words → average embeddings
- Output: one center word
- Good for: faster, smoother, frequent words

### Skip-gram
- Input: one center word
- Output: many context words (as many pairs)
- Good for: rare words, better semantic detail (often), more training pairs

### Both
- Same parameter shapes: \(W\in\mathbb{R}^{V\times d}\), \(U\in\mathbb{R}^{d\times V}\)
- Full softmax = expensive; negative sampling = practical
- Embeddings = rows of \(W\) (commonly)

---

