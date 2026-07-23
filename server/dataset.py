"""
Server-side reproduction of the browser's dataset pipeline.

The leaderboard is only fair if the referee evaluates every student on exactly
the same data the browser showed them. This module is a faithful port of
`src/lib/rng.ts`, `src/lib/mnist.ts` and `src/lib/dataset.ts` — same PRNG, same
shuffle, same downsample, same split — so a given seed produces a byte-identical
train/test partition on both sides.

`verify_parity.mjs` next to this file checks that claim against the real
TypeScript implementation.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MNIST_DIR = BASE_DIR.parent / "public" / "mnist"

RAW_SIZE = 28
RAW_PIXELS = RAW_SIZE * RAW_SIZE
FEATURE_SIZE = 14
FEATURE_PIXELS = FEATURE_SIZE * FEATURE_SIZE
NUM_CLASSES = 10

STANDARD_SAMPLES = 1200
STANDARD_TEST_SIZE = 0.2

_MASK = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    """JS Math.imul — 32-bit multiply, keeping the low 32 bits."""
    return (a * b) & _MASK


def make_rng(seed: int):
    """mulberry32, bit-identical to src/lib/rng.ts.

    Everything is kept in unsigned 32-bit form. JS applies ToInt32 at each
    bitwise operator, which is the same bit pattern; the one place the sum can
    exceed 32 bits is masked explicitly below, exactly where `^` would do it.
    """
    a = seed & _MASK

    def rng() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & _MASK
        t = _imul(a ^ (a >> 15), 1 | a)
        t = ((t + _imul(t ^ (t >> 7), 61 | t)) & _MASK) ^ t
        return ((t ^ (t >> 14)) & _MASK) / 4294967296

    return rng


@lru_cache(maxsize=1)
def load_mnist_raw():
    """The same two flat uint8 files the browser fetches from /mnist/."""
    images = np.fromfile(MNIST_DIR / "images28.bin", dtype=np.uint8)
    labels = np.fromfile(MNIST_DIR / "labels.bin", dtype=np.uint8)
    count = len(labels)
    if len(images) != count * RAW_PIXELS:
        raise RuntimeError(
            f"MNIST files disagree: {len(images)} image bytes for {count} labels "
            f"(expected {count * RAW_PIXELS})."
        )
    return images.reshape(count, RAW_PIXELS), labels, count


def downsample(images: np.ndarray) -> np.ndarray:
    """2x2 block average, 28x28 -> 14x14. Matches downsampleImage()."""
    n = images.shape[0]
    blocks = images.reshape(n, FEATURE_SIZE, 2, FEATURE_SIZE, 2).astype(np.float64)
    return blocks.mean(axis=(2, 4)).reshape(n, FEATURE_PIXELS)


def shuffled_indices(seed: int, count: int) -> list[int]:
    """Fisher-Yates driven by mulberry32 — the same order generateDataset() uses."""
    rng = make_rng(seed)
    indices = list(range(count))
    for i in range(count - 1, 0, -1):
        j = int(rng() * (i + 1))
        indices[i], indices[j] = indices[j], indices[i]
    return indices


def build_dataset(seed: int):
    """Returns (X_train, y_train, X_test, y_test) for a seed.

    Mirrors generateDataset(): shuffle every index, take the first
    STANDARD_SAMPLES, and slice the leading 20% off as the test set.
    """
    images, labels, count = load_mnist_raw()
    n = min(STANDARD_SAMPLES, count)
    order = shuffled_indices(seed, count)[:n]

    features = downsample(images[order])
    y = labels[order].astype(np.int64)

    test_count = round(n * STANDARD_TEST_SIZE)
    return (
        features[test_count:], y[test_count:],   # train
        features[:test_count], y[:test_count],   # test
    )
