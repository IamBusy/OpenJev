# Names and versions

Keep the project, model, software version and training revision distinct.

| Item | Public name / example | Meaning |
| --- | --- | --- |
| Project and GitHub repository | OpenJev | Research toolkit and source code |
| Model | OpenJev-0.6B | Model family and nominal backbone parameter size |
| Hugging Face repository | IamBusy/OpenJev-0.6B | Stable model download address |
| Python package | openjev | Installable software |
| Public Python class | `from openjev import OpenJevModel` | Typed-decision model loader |
| Current model command | `openjev-model` | Download, inference, training and evaluation |
| Software release | v0.3.2 | Changes to the toolkit and packaging |
| Trained model revision | 0.3.0 | The existing development-selected v0.3 checkpoint |
| Exact artifact identity | Repository commit and SHA-256 hashes | Reproduction and verification |

`0.6B` refers to the base model's nominal size, not the size of the adapter
archive. LoRA and the scalar head contain about 1.15 million trained parameters;
the base weights are needed to use the resulting model.

Architecture terms such as candidate branches belong in implementation names and
model cards. Experiment labels such as v0.1/v0.2/v0.3 belong in protocols, training
runs and reports. They are not appended to the stable public model repository name.

## Compatibility

The former Hugging Face repository redirects to the canonical name. Existing
published GitHub releases and assets remain available at their historical URLs.
The current release has a canonically named `OpenJev-0.6B.zip` bundle.

`openjev-branch` remains an alias for `openjev-model`; existing
`openjev.branch_model.BranchDecision` imports remain valid. Inference prefers
`artifacts/openjev-0.6b` and falls back to the earlier local checkpoint directory
when necessary. The historical MiniLM `openjev` command is retained for old
experiments; current-model documentation uses `openjev-model` consistently.

The inference response reports `model: OpenJev-0.6B` and `model_version: 0.3.0`
separately. Renaming and packaging do not retrain or merge the weights. Historical
result records keep their original experiment identities and measured values.
