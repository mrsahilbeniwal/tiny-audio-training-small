# Tiny Audio ASR Training

**Author:** Sahil Beniwal

A Speech-to-Text model trained using the [Tiny Audio](https://github.com/alexkroman/tiny-audio) framework on E2E Networks.

## Architecture

```
Audio (16kHz) → Whisper Encoder (frozen) → MLP Projector (trained) → SmolLM3-3B (frozen) → Text
```

| Component | Model | Parameters | Status |
|-----------|-------|------------|--------|
| Audio Encoder | openai/whisper-large-v3-turbo | 800M | Frozen |
| Projector | MLP | 11.7M | Trained |
| Language Model | HuggingFaceTB/SmolLM3-3B | 3B | Frozen |

## Training Configuration

| Setting | Value |
|---------|-------|
| GPU | NVIDIA H100 80GB HBM3 |
| Dataset | speechbrain/LoquaciousSet (small) |
| Train Samples | 1,000 |
| Max Steps | 500 |
| Batch Size | 8 |
| Learning Rate | 3e-4 |

## Training Results

| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 100 | 3.08 | 3.17 |
| 300 | 0.50 | 0.81 |
| 500 | 0.10 | 0.76 |

## Files

- `train_tiny_audio.ipynb` - Training notebook with outputs
- `README.md` - This file

## References

```bibtex
@software{kroman2025tinyaudio,
  author = {Kroman, Alex},
  title = {Tiny Audio: Train Your Own Speech Recognition Model in 24 Hours},
  year = {2025},
  url = {https://github.com/alexkroman/tiny-audio}
}

@misc{loquaciousset2024,
  author = {{SpeechBrain Team}},
  title = {LoquaciousSet: A Large-Scale Speech Recognition Dataset},
  url = {https://huggingface.co/datasets/speechbrain/LoquaciousSet}
}
```

## License

Apache 2.0 - See the [Tiny Audio repository](https://github.com/alexkroman/tiny-audio) for details.
