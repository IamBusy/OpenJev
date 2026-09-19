# OpenJev-Branch-v0.3 model card

**Status:** experimental research checkpoint, released 2026-09-19.
[Code](https://github.com/IamBusy/OpenJev) ·
[Release](https://github.com/IamBusy/OpenJev/releases/tag/v0.3.0)

## Model and use

The checkpoint contains a Qwen3-0.6B attention-projection LoRA (1,146,880 trainable
parameters), an independent 1,024-parameter scalar head and per-primitive
calibration. The base model is separate, pinned to revision
`c1899de289a04d12100db370d81485cdf75e47ca` of `Qwen/Qwen3-0.6B`.
Use OpenJev's `BranchDecision` loader; this is not a standalone Transformers
`AutoModel` or a normal text-generation LoRA. The scalar head is required.

The head starts from the pretrained Yes-minus-No embedding direction and is
trained independently. Prediction bypasses the vocabulary projection and does
not autoregressively decode answers. Prefix caches live within a request or a
training state group and never cross optimizer updates. Candidates are scored
independently and normalized within a question.

Intended use is local research into typed decisions, candidate scoring and shared
prefix computation. Training/evaluation include English and Chinese simulated
states, but broad multilingual capability is not established. The model has no
validated role in consequential autonomous decisions.

## Data, training and evaluation

725 training judgments from 486 state groups; 176 development, 167 calibration,
405 fresh test and 448 prior regression judgments. The 60-question reference set
is a fixed subset of fresh test. Sources include BANKING77, TweetEval sentiment,
BoolQ, ARC-Easy and original refund/incident rule scenarios. Access and shipping
scenario families are held out. Rules compute scenario labels; missing evidence
is an explicit candidate, not a fabricated probability target.

DeepSeek rendered 48 selected worlds and independently reconstructed their facts.
27 renderings passed verification; rejected cases use templates. The same teacher
family verified its own renderings, so this is not independent human review.
Canonical facts and accepted narratives are in the public scenario snapshot.
Public pretraining contamination and template-specific learning remain possible.

Two epochs, seed 43; 244 updates. Selected solely by development NLL (1.0094 to
0.41665). Training took approximately 636 seconds on an M3 Pro with 36 GB RAM.
The configuration and hashes are recorded in the code repository's
`configs/branch-v03.json` and `reports/v03/TRAINING_SELECTION.json`.

On the fixed 60-question reference subset: initial branch model 40/60, trained
branch model 45/60, v0.2 Qwen 39/60, DeepSeek 59/60. On 16 matched candidate-reversal
checks, v0.3 was consistent in 16/16 cases; an additional 32 fresh cases also
passed. These are bounded checks, not universal correctness claims.

## Limitations and regressions

Some banking, access-policy and shipping tasks regress relative to v0.2. Access
clearance-gap accuracy is 6.67% on its small fresh subset; shipping-action accuracy
is 20%. Scores can be confidently wrong. Calibration applies to the measured
mixture only. Independent candidates cannot directly interpret other candidates,
which limits set-relative criteria such as “none of the above.”

Architecture and training data changed together, so accuracy gains are not an
architecture-only ablation. This is one training seed. A 255-candidate smoke test
checks execution and output shape, not general large-label performance.

Warm synthetic timing on the M3 Pro: long state/8 questions/4 candidates about
0.70 s versus v0.2's 1.38 s; short state/1 question about 104 ms versus 55 ms.
See `RESULTS.md` in the model bundle or `reports/v03/RESULTS.md` in the source.
No speed or accuracy parity with Jev is claimed. This is supervised training,
not a reproduction of TypeSafe's unpublished RLCD algorithm.

## Distribution and licenses

The adapter/head bundle and code are distributed under Apache-2.0. The Qwen base
is Apache-2.0 and downloaded separately. Dataset texts retain their original
licenses and attribution, including BoolQ CC BY-SA 3.0 and ARC CC BY-SA 4.0;
see `THIRD_PARTY.md`. The model bundle includes no raw benchmark corpus, private
provider traces or credentials. No legal conclusion about downstream use is
implied by model behavior.

The trained adapter and scalar head are available on
[Hugging Face](https://huggingface.co/IamBusy/OpenJev-Branch-v0.3) and in the original
GitHub Release ZIP. Both use identical trained tensor bytes. See
[Hub loading instructions](HUGGING_FACE.md). The original ZIP SHA-256 is pinned in
`src/openjev/release_assets.json`; each bundle has its own file manifest.
