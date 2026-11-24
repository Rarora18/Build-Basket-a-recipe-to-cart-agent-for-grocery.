# Basket

A meal request goes in. A priced grocery cart comes out.

> high-protein dinners for 2 this week, under $60, no pork.

Basket plans the dinners, matches each ingredient to a SKU, swaps in a substitute when the preferred product is out of stock, and solves a small integer program so the cart stays under budget and picks up flyer deals. The request above comes back at **$59.63**. Lentils are out of stock, so their grams are folded into the kidney beans already on the list, and one dinner is dropped so the budget holds.

## How it is put together

1. **Planner.** A parser pulls servings, budget, and diet flags out of the sentence. An optional LLM call (`OPENAI_API_KEY`) may choose recipe ids; without a key, a protein-per-dollar heuristic does it. Either way the recipes have to exist in the catalog.
2. **Hybrid retrieval.** Okapi BM25 and a hashed character-trigram embedding are fused per query. Lexical weight goes up when one product dominates the BM25 scores, and the embedding channel takes typos and synonyms (`coriander` → cilantro, `chicken thghs` → chicken thighs). With `DATABASE_URL` set, the same fusion runs on Postgres full-text rank and pgvector cosine.
3. **Substitutes.** Item2vec here is a positive-PMI profile of each SKU (the matrix skip-gram factorizes). A LightGBM ranker scores candidate swaps from that cosine, PMI, aisle, nutrition, price, and name overlap. The group id is the training label, not a feature. Anchors used at eval time are held-out ingredients.
4. **Optimizer.** PuLP / CBC picks one SKU and a pack count per ingredient under the budget. If the plan cannot fit, the lowest protein-per-dollar dinner is dropped and the model is solved again.
5. **Service.** FastAPI plus a single-page cart, packaged with Docker and pgvector.

## Measured

On the 200 labelled ingredient → SKU queries in `data/seed/labels.jsonl`:

| Channel | Hit@5 | Precision@5 |
| --- | ---: | ---: |
| BM25 | 0.870 | 0.514 |
| Embeddings | 0.995 | 0.593 |
| Hybrid | 1.000 | 0.597 |

Precision@5 is the share of the top 5 SKUs that belong to the labelled ingredient. Three packs are labelled per ingredient, so 0.60 is the ceiling. The hybrid gain is concentrated on synonyms, typos, and paraphrases; clean queries already saturate.

On 1,598 held-out basket anchors, candidates restricted to the same aisle:

| Ranker | Hit@1 |
| --- | ---: |
| Same-aisle popularity | 0.470 |
| Item2vec nearest neighbour | 0.732 |
| LightGBM ranker | 1.000 |

The ranker beats popularity by **112.8%** relative hit@1. Item2vec alone, which never sees the group label, is up 55.8%. Nine diet and budget requests produced **0** constraint violations. Local planning latency is about **88 ms** p50 and **94 ms** p95, and **$0** per request when the heuristic planner is used.

Numbers are written by `python scripts/eval_all.py` to `evals/latest.json`.

## Resume bullet

Built a grocery agent that turns a natural-language meal request into an optimized cart. Hybrid BM25 + embedding retrieval, with the same fusion on pgvector, raised ingredient→SKU precision@5 from 0.514 to 0.597 on 200 labelled queries. A co-purchase substitution ranker beat same-aisle popularity by 112.8% hit@1 on held-out ingredients.

The baskets follow the Instacart order schema and are synthetic, so this bullet does not claim a model trained on the 3M-order Kaggle dump. `scripts/import_instacart.py` loads that dump when you have it.

## Run it

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python scripts/train.py   # already checked in; rerun after regenerating data
.venv/bin/python -m uvicorn basket.api:app --port 8000
```

Open http://localhost:8000.

Postgres and pgvector:

```bash
docker compose up --build
```

That starts `pgvector/pgvector:pg16`, loads the catalog, recipes, and baskets, and serves the API on port 8000 with `DATABASE_URL` set. The dev database password is `basket`.

Regenerate the seed, retrain, test, and re-eval:

```bash
make seed train test eval
```

## Data

`data/seed/` is a Canadian-style catalog (PC, No Name, Selection, and a few house brands), 30 dinners, 6,500 baskets, weekly flyer markdowns, and stockouts. Prices are synthetic.

| File | What it is |
| --- | --- |
| `ingredients.jsonl`, `products.jsonl` | Catalog, nutrition, allergens, price, stock, deal |
| `recipes.jsonl` | Dinners aligned to those ingredients |
| `orders.jsonl` | Baskets in Instacart shape: order id, sku list, train/test split |
| `labels.jsonl` | 200 ingredient → SKU queries (clean, paraphrase, synonym, typo) |

To point the same code at the public sources:

- `scripts/import_instacart.py` reads the Kaggle Market Basket CSVs.
- `scripts/import_off.py` streams an Open Food Facts dump and keeps `en:canada`.
- `scripts/import_recipenlg.py` reads a RecipeNLG CSV.

Those dumps are not in the repo.

## Layout

```
basket/     planner, retrieval, item2vec, ranker, optimizer, API
web/        cart UI
sql/        Postgres schema, including the vector column
scripts/    seed, train, eval, importers
evals/      latest metrics
artifacts/  item2vec vectors and the LightGBM model
```
