# Public-image protocol v0.1

Declared before fitting public-image models or evaluating test labels.

## Data

Oxford-IIIT Pet is CC BY-SA 4.0. Use the pinned timm Hub mirror, and verify every
mirror ID/species/split against the original Oxford annotation archive. Within
each breed, sort original IDs by the specified SHA256 key. Select 40 train,
10 development, and 10 calibration images from the official trainval set;
20 official test images per breed remain test (2,960 images total).

CLEVR-4 10k v1 is CC BY 4.0. Read original images/annotations using verified ZIP
ranges, preserving archive identity, member CRC and individual SHA256. Sort colors
and shapes lexically. Pairs with (color_index+shape_index)%5 == 0 are held out of
training, development AND calibration. Select 12/2/2 images for each other pair
from official train; select four images per pair from official val as our test.
This yields 960/160/160/400 images. Report seen and unseen compositions separately.
This is a bounded subset, not the official full-dataset benchmark.

Resize only the longest side to at most 256 and save lossless PNG. Keep all views
and derived questions for each original image in its assigned split.

## Frozen visual representation

Use DINOv2-small, pinned to ed25f3a31f01632728cabb09d1542f84ab7b0056.
Use its checked-in image processor (shortest edge 256, center crop 224,
ImageNet normalization), CLS representation, frozen float32 weights. This is a
pretrained vision backbone; upstream pretraining overlap cannot be ruled out.

## Heads

Pets: train a 37-way linear breed head; derive species and Boolean subset
probabilities by summing the same breed distribution. General language grounding
is not learned by this head.

CLEVR-4: compare independent color/shape heads; a 100-way joint linear head; and a
low-rank binding head. The latter adds an image-conditioned, rank-four residual
interaction between learned color and shape embeddings to unary logits, then
normalizes the joint table. Low-rank interaction is a known modeling technique,
not a claim of original research. Its unseen-composition behavior is measured.
The main comparison uses the same frozen feature matrix.

Training: seeds 17, 23, 42; AdamW, lr 0.005, weight decay 0.001, 120 epochs,
batch size 256. Feature mean/std fitted on train only. Checkpoint selection by
development NLL. One scalar temperature fitted on calibration only for each head;
temperature is applied before joint normalization or to the independent logits,
never independently to derived query answers. Report every seed.

## Metrics and limits

Accuracy, NLL, Brier, ECE with bin counts; breed/species and color/shape/joint metrics;
seen/unseen composition results; probabilities for Boolean compound queries.
Report validation-selected and calibrated results without test-driven retuning.

Synthetic exact-posterior results and real-photo label-based results must remain
separate. A public-image class label is not a ground-truth uncertainty distribution.
Do not claim that the synthetic-scene CNN works on pet photos, or that the
frozen-feature public heads are general-purpose VLMs.

## Publication

Publish resized selected images with original source IDs, attribution, exact
transform description and source licenses. Keep Oxford derivatives CC BY-SA 4.0,
CLEVR-4 derivatives CC BY 4.0, and original synthetic scenes Apache-2.0, in separate
dataset configurations with separate notices. Original code remains Apache-2.0.
