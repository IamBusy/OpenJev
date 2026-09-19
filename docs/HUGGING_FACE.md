# Hugging Face distribution

The trained v0.3 model is published at
[IamBusy/OpenJev-0.6B](https://huggingface.co/IamBusy/OpenJev-0.6B).
The model parameters are the same development-selected checkpoint as the original
GitHub v0.3.0 release. OpenJev package v0.3.2 provides the stable public name and loading entry point.

## Load the trained decision model

```bash
pip install 'openjev[qwen] @ git+https://github.com/IamBusy/OpenJev.git@v0.3.2'
```

```python
from openjev import OpenJevModel

model = OpenJevModel.from_pretrained("IamBusy/OpenJev-0.6B")
result = model.predict(
    state="The recorded color is red.",
    questions={
        "color": {
            "type": "choice",
            "instructions": "Which candidate agrees with the recorded color?",
            "criteria": {"red": "The color is red.", "blue": "The color is blue."},
        }
    },
)
print(result["answers"])
```

The first load downloads the adapter, custom head, calibration and pinned Qwen
base into the Hugging Face cache. Subsequent loads reuse the cache. No checkout
working directory or manually arranged base-model folder is required. Use
`revision="<model repository commit>"` to pin the model snapshot, and
`local_files_only=True` after downloading to run without network access.

The loader verifies the bundle's file manifest. It automatically applies the
published calibration; pass `temperatures={}` to request uncalibrated output.
Calibration remains specific to the measured dataset mixture. The direct
constructor retains its explicit calibration behavior.

## Why an adapter release is complete for this training method

LoRA training updates adapter matrices while freezing the base weights. Publishing
those matrices, their configuration and the exact base revision is a supported
PEFT distribution format. A merged full checkpoint is another option, with a much
larger download. See the [official PEFT checkpoint guide](https://huggingface.co/docs/peft/developer_guides/checkpoint).

OpenJev also trains a scalar scoring head outside the LoRA module. Therefore the
adapter alone is insufficient: `head.safetensors`, `openjev_config.json`, the
calibration file and OpenJev's candidate-branch inference implementation are part
of the usable release. The Hub repository includes those files. It is a typed
decision model, not an ordinary chat-generation adapter. Loading only the PEFT
adapter with a generic text-generation loader does not reproduce OpenJev.

The Hub layout puts `adapter_model.safetensors` and `adapter_config.json` at the
repository root; the older GitHub ZIP keeps them under `adapter/`. Both layouts
contain identical trained tensor bytes and are supported by OpenJev v0.3.1.

Public model names and compatibility aliases are defined in [NAMING.md](NAMING.md).
