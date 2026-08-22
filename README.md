# HDCbase

Tiny Hyperdimensional Computing simulator.

```
main.py          run this to see it work
hdc/
  vectors.py     makes random hypervectors
  ops.py         bind, bundle, similarity
```

## The idea

A hypervector is just a random array of -1s and 1s (10,000 of them by
default). Random ones are always very different from each other. Three
functions let you combine/compare them:

- `bind(a, b)` — combine two vectors into something new (different from both)
- `bundle(a, b)` — combine two vectors into something similar to both
- `similarity(a, b)` — how alike two vectors are, -1 to 1

Run `python main.py` to see all three in action.

## Where to go next

- Give things names, e.g. `apple = random_vector()`, and keep them in a
  plain `dict` so you can look them up later.
- Try `bind`-ing a "key" and a "value" (like `color` and `red`), then
  `bind` again with the same key to get the value back.
- Try encoding a word as letters: make a vector per letter, `bind` each
  letter to its position, `bundle` them all together.
