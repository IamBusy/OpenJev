# Sources and attribution

OpenJev is an independent research implementation inspired by TypeSafe's Jev.
It is not affiliated with or endorsed by TypeSafe. It does not contain Jev weights
or claim to reproduce TypeSafe's unpublished architecture or RLCD training.

## Model

- `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, Apache-2.0.
- Model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Sentence-BERT: Nils Reimers and Iryna Gurevych (2019),
  https://aclanthology.org/D19-1410/
- The backbone is frozen. OpenJev trains a new shared residual scoring head.
  The original model's pretraining data can overlap benchmark subject matter;
  this experiment establishes post-training isolation only.

## Public data (licenses remain separate from the code license)

| Source | Version | License / attribution |
| --- | --- | --- |
| BANKING77 | `57ec275d8078af65b7731c2a98be812d844a6d6b` | CC BY 4.0; PolyAI, Casanueva et al., 2020 |
| CLINC150 | `828f8093932c8fe6ca7936c3d2e52903b1c523de` | CC BY 3.0; Larson et al., 2019 |
| TweetEval sentiment | `4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66` | CC BY 3.0 for the sentiment task, as documented in the Hugging Face dataset card; Rosenthal et al., 2017, and Barbieri et al., 2020 |

- BANKING77: https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data
- BANKING77 license: https://github.com/PolyAI-LDN/task-specific-datasets/blob/57ec275d8078af65b7731c2a98be812d844a6d6b/LICENSE
- CLINC150: https://github.com/clinc/oos-eval/tree/828f8093932c8fe6ca7936c3d2e52903b1c523de
- TweetEval: https://github.com/cardiffnlp/tweeteval/tree/4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66
- Task-specific license documentation: https://huggingface.co/datasets/cardiffnlp/tweet_eval#licensing-information
- TweetEval's umbrella README refers users to original dataset and platform terms;
  the sentiment license must not be generalized to its other tasks. The default
  source-code distribution fetches these public sources directly; it does not
  relabel their texts as Apache-2.0 code.

`data/SOURCE_LOCK.json` records download locations, pinned revisions and SHA-256
checksums. Original license files are downloaded into `data/raw` where provided.
Processed data changes include normalized-text deduplication, split assignment,
candidate descriptions, ordered-label formatting and binary proposition views.
Source record IDs and modification provenance are retained.

## Synthetic data

- Local urn simulator: original generator code under the project Apache-2.0
  license. Targets are exactly computed draw probabilities, not teacher opinions.
- DeepSeek paraphrases: generated using the user's authorized DeepSeek account,
  based only on public *training* records. Keep upstream attribution and license
  information for each source; do not describe the mixed dataset as one uniform
  Apache-2.0 corpus. Labels are generated and reviewed by DeepSeek, not independently
  human-verified. Raw requests, responses and usage are retained locally without
  credentials. Provider API aliases can change: saved snapshots are the exact
  reproducibility source, not a claim that future live calls return identical text.

## v0.2 additions

- Qwen/Qwen3-0.6B, revision c1899de289a04d12100db370d81485cdf75e47ca,
  Apache-2.0: https://huggingface.co/Qwen/Qwen3-0.6B
- BoolQ evaluation only, revision 35b264d03638db9f4ce671b711558bf7ff0f80d5,
  CC BY-SA 3.0: https://huggingface.co/datasets/google/boolq
- ARC-Easy evaluation only, revision 210d026faf9955653af8916fad021475a3f00453,
  CC BY-SA 4.0: https://huggingface.co/datasets/allenai/ai2_arc
- Clark et al. (2019), BoolQ: https://arxiv.org/abs/1905.10044
- Clark et al. (2018), ARC: https://arxiv.org/abs/1803.05457

The new evaluation texts retain their share-alike source licenses. They are not
included in the Qwen LoRA training corpus. The first Qwen adapter uses only the
recorded BANKING77 and TweetEval training subsets; the older DeepSeek paraphrases
are excluded. New DeepSeek calls are benchmark predictions, not training labels.
See data/v02/SOURCE_LOCK.json and reports/v02/manifest.json.

## v0.3 additions and changes

The v0.3 branch experiment uses the same pinned Qwen base. It now uses selected
public BoolQ and ARC training records in addition to their evaluation records.
Those datasets retain CC BY-SA 3.0 and CC BY-SA 4.0, respectively. The preceding
evaluation-only statement describes v0.2, not this new experiment.

New software-state scenarios, rule oracles and templates are original project
material. DeepSeek renders some narratives from original structured facts and
independently reconstructs those facts; it does not provide the gold outcome.
Raw provider responses contain no authorization headers and stay outside Git.
Public and generated data must keep their distinct provenance and license terms.

Shared-prefix engineering uses the installed Transformers DynamicCache/Qwen3
implementation, with explicit forward/gradient equivalence tests. No third-party
project implementation or proprietary TypeSafe code was copied.

## Method references

- TypeSafe introduction: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- TypeSafe primitives: https://docs.typesafe.ai/primitives
- Temperature scaling: Guo et al. (2017), https://proceedings.mlr.press/v70/guo17a.html
- Related open work: https://github.com/Knowledgator/GLiClass,
  https://github.com/TianyuCodings/NanoJev, https://github.com/TheoLeeCJ/SemIf

The implementation here was written independently. This pilot makes no
performance-superiority claim over these projects or over Jev.

## Included examples

`examples/data_records_v01.json` contains a few source-attributed example records
to illustrate the schema. Their source texts retain the dataset licenses above;
they are not relicensed as project code. Full source corpora and processed splits
are downloaded/reconstructed locally and excluded from Git.
