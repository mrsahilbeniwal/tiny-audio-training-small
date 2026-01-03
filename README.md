# 🎤 Tiny Audio ASR Training

Training a Speech-to-Text model using the [Tiny Audio](https://github.com/alexkroman/tiny-audio) framework on E2E Networks GPU cloud.

## Architecture

```
Audio (16kHz) → Whisper Encoder (frozen) → MLP Projector (trained) → SmolLM3-3B (frozen) → Text
```

| Component | Model | Parameters | Status |
|-----------|-------|------------|--------|
| Audio Encoder | openai/whisper-large-v3-turbo | ~800M | ❄️ Frozen |
| **Projector** | MLP | **11.7M** | 🔥 **Trained** |
| Language Model | HuggingFaceTB/SmolLM3-3B | 3B | ❄️ Frozen |

**Only 0.32% of parameters are trained!**

## Training Details

| Setting | Value |
|---------|-------|
| **GPU** | NVIDIA H100 80GB HBM3 |
| **Cloud** | E2E Networks |
| **Dataset** | speechbrain/LoquaciousSet (small) |
| **Train Samples** | 1,000 |
| **Max Steps** | 500 |
| **Training Time** | ~18 minutes |
| **Framework** | PyTorch 2.8.0 + Transformers |

## Training Loss

| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 100 | 3.08 | 3.17 |
| 200 | 2.54 | 3.16 |
| 300 | 0.50 | 0.81 |
| 400 | 0.14 | 0.73 |
| 500 | **0.10** | **0.76** |

## Sample Results

**Ground Truth:**
```
THESE ARE REFORMS THAT WILL DISCIPLINE AND CONSTRAIN THE EXERCISE OF POWER 
BY THE GOVERNMENT AND ANY OTHER ECONOMIC OR POLITICAL ACTOR FOR GENERATIONS TO COME
```

**Model Prediction:**
```
These are reforms that will discipline and constrain the exercise of power 
by the government and any other economic or political actor for generations to come
```

✅ **Perfect transcription!**

## Quick Start

```python
from src.asr_config import ASRConfig
from src.asr_modeling import ASRModel

# Load model
config = ASRConfig(
    audio_model_id="openai/whisper-large-v3-turbo",
    text_model_id="HuggingFaceTB/SmolLM3-3B",
    projector_type="mlp",
)
model = ASRModel(config)

# Transcribe
audio_array = load_your_audio()  # 16kHz numpy array
inputs = model.feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")
output = model.generate(input_features=inputs.input_features.to(model.device).to(model.dtype))
transcription = model.tokenizer.decode(output[0], skip_special_tokens=True)
print(transcription)
```

## Files

- `train_tiny_audio.ipynb` - Complete training notebook with outputs
- `README.md` - This file

## References

### Tiny Audio Framework
```bibtex
@software{kroman2025tinyaudio,
  author = {Kroman, Alex},
  title = {Tiny Audio: Train Your Own Speech Recognition Model in 24 Hours},
  year = {2025},
  url = {https://github.com/alexkroman/tiny-audio}
}
```

### LoquaciousSet Dataset
```bibtex
@misc{loquaciousset2024,
  author = {{SpeechBrain Team}},
  title = {LoquaciousSet: A Large-Scale Speech Recognition Dataset},
  url = {https://huggingface.co/datasets/speechbrain/LoquaciousSet}
}
```

### Base Models
- **Whisper**: [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
- **SmolLM3**: [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)

## License

This project uses the Tiny Audio framework. See the [original repository](https://github.com/alexkroman/tiny-audio) for license details.

---

*Trained on E2E Networks with NVIDIA H100 GPU*
