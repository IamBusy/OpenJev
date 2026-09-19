"""Finite scene semantics and a transparent observation model."""

from itertools import product

import numpy as np

SLOTS = ("left", "center", "right")
ATTRIBUTES = ("color", "shape")
VALUES = {"color": ("blue", "red"), "shape": ("circle", "square")}
WORLDS = np.asarray(list(product((0, 1), repeat=6)), dtype=np.int64)
WORLD_COUNT = len(WORLDS)
EDGES = {
    "local": ((0, 1), (2, 3), (4, 5)),
    "chain": ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)),
    "cross": ((0, 4), (1, 5), (0, 3), (2, 5)),
}


def normalize_prior(prior):
    p = np.asarray(prior, dtype=np.float64)
    if p.shape != (WORLD_COUNT,) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("prior must contain 64 finite, nonnegative weights")
    if p.sum() <= 0:
        raise ValueError("prior must have positive total mass")
    return p / p.sum()


def sample_prior(rng, family):
    if family not in EDGES:
        raise ValueError(f"Unknown dependency family: {family}")
    spin = 2 * WORLDS - 1
    energy = spin @ rng.uniform(-0.35, 0.35, 6)
    for i, j in EDGES[family]:
        weight = rng.uniform(0.8, 1.5) * rng.choice((-1, 1))
        energy += weight * spin[:, i] * spin[:, j]
    return normalize_prior(np.exp(energy - energy.max()))


def posterior(prior, observed, sensor_error=0.08):
    """Posterior given observable sensor symbols, never the hidden true world."""
    p = normalize_prior(prior)
    obs = np.asarray(observed)
    if obs.shape != (6,) or not np.isin(obs, (-1, 0, 1)).all():
        raise ValueError("observed must contain six values in {-1, 0, 1}")
    if not 0 < sensor_error < 0.5:
        raise ValueError("sensor_error must be strictly between 0 and 0.5")
    likelihood = np.where(
        obs[None, :] == -1,
        1.0,
        np.where(WORLDS == obs[None, :], 1 - sensor_error, sensor_error),
    ).prod(axis=1)
    return normalize_prior(p * likelihood)


def marginalize(p):
    p = np.asarray(p, dtype=np.float64)
    return p @ WORLDS


def factorize(p):
    """Discard dependencies while retaining the supplied distribution's marginals."""
    marginals = marginalize(p)
    return np.where(WORLDS == 1, marginals[..., None, :], 1 - marginals[..., None, :]).prod(axis=-1)


def describe_world(index):
    bits = WORLDS[int(index)]
    return {
        slot: {
            attribute: VALUES[attribute][int(bits[2 * i + j])]
            for j, attribute in enumerate(ATTRIBUTES)
        }
        for i, slot in enumerate(SLOTS)
    }
