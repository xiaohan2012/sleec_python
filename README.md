# sleec_python

Python implementation of "Sparse Local Embeddings for Extreme Multi-label Classification, NIPS, 2015"

# main scripts

- `main.py`: the whole pipeline

# variations

list of variations from the original paper:

1. cosine distance to build the kNN graph for embedding learning
2. Alternating Least Sqaure to learn low-rank embedding `Z`
   - without `l1` regularization
   - the paper uses Singular Value Projection
3. Ridge regression to optimize the linear regressor
  - without `l1` regularization
  - the papers uses ADMM with `l1`

# update
  - **2017-11-01**: for bibtex dataset, achieved p1, p3 and p5 are 0.5964 (0.6532), 0.3455 (0.3973) and 0.2461 (0.2889) (`(..)` is score by the paper)