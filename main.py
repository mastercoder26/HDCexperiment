from hdc import random_vector, bind, bundle, similarity

apple = random_vector()
banana = random_vector()

# random vectors are unrelated
print("apple vs banana:               ", similarity(apple, banana))

# bind() makes something different from both inputs
combo = bind(apple, banana)
print("bind(apple, banana) vs apple:  ", similarity(combo, apple))

# bundle() makes something similar to both inputs
mix = bundle(apple, banana)
print("bundle(apple, banana) vs apple:", similarity(mix, apple))
