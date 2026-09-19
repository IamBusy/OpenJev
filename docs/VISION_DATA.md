# Visual data and supervision

| Configuration | Records | Provenance | License |
|---|---:|---|---|
| synthetic | 8,192 | Original renderer and exact finite observation model | Apache-2.0 |
| pets | 2,960 | Oxford-IIIT Pet, original official IDs/splits verified | CC BY-SA 4.0 |
| clevr4 | 1,680 | Official CLEVR-4 10k-v1 archive | CC BY 4.0 |

The [published dataset](https://huggingface.co/datasets/IamBusy/OpenJev-Vision-Research-v0.1)
has separate configurations; no single Apache license is asserted over the mixture.
Original owners retain rights in public images. See its LICENSES.md and the
repository's [attribution](../THIRD_PARTY.md).

## Public sources

Oxford-IIIT Pet: https://www.robots.ox.ac.uk/~vgg/data/pets/
by Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman and C. V. Jawahar.
Mirror: timm/oxford-iiit-pet, revision 089695c834a7deb60505b7cc506672db1c31a6aa.
The official annotation archive's SHA256 is
52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91.

CLEVR-4: https://www.robots.ox.ac.uk/~vgg/data/clevr4/
by Sagar Vaze, Andrea Vedaldi and Andrew Zisserman.
Use clevr_4_10k_v1.zip, size 3,797,490,816 bytes, ETag
"653ba205-e2591c80". The loader reads selected archive ranges and verifies ZIP CRC
and individual source-image SHA256. It records source metadata without claiming
to hash the complete archive when it has not downloaded it.

Public transformations are deterministic subset selection, RGB conversion,
longest-side resize to <=256 with Lanczos, PNG encoding, and questions derived
from annotations. No teacher-generated label is treated as ground truth.
Per-file and dataset manifests are included with the release.

## Split rules

Pet split sizes per breed: 40 train, 10 development, 10 calibration from official
trainval, and 20 official test images. Original IDs are sorted by a fixed SHA256.

CLEVR-4 colors and shapes use lexicographic indices. The 20 pairs with
(color_index+shape_index)%5 == 0 are absent from train/dev/calibration.
For every other pair select 12 train, 2 dev, 2 calibration original train images.
Test has 4 official validation images for each of all 100 pairs.

Exact public source-byte and resized-image duplicate checks run before export.
These checks do not establish absence of near duplicates or pretraining overlap.
All views/questions derived from the same original image/episode remain together.

## Meaning of the synthetic probabilities

Each original episode samples a six-bit world from a supplied prior.
Sensor noise flips each attribute with probability 0.08. Occlusion masks whole
objects. The renderer sees only the noisy visible symbols. The oracle conditions
on those observable symbols and marginalizes hidden values.

The hidden true world's label is separate. Replacing the posterior with that
world's one-hot label would incorrectly train certainty from unavailable evidence.
Observation metadata is privileged supervision for the evidence baseline; it
is never an input to normal image inference.

There are 36 derived Boolean questions per image (294,912 targets), plus five
annotation-derived questions per public image (23,200 targets). These are
318,112 derived judgments, not that many independent observations.

## Independent reconstruction

The source manifests retain generator/data hashes, mirror revision, original
image IDs, rendering seeds, sensor assumptions, held-out pairs and output hashes.
The public Hub export uses image-Parquet with embedded images, so it can be
loaded without local filenames. The three JSON-string fields preserve
question semantics, target distributions and provenance.
