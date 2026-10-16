"""slm-forge: fine-tuning and factual-reliability evaluation for small language models.

The core package (everything except ``eval.generate_hf``, ``train.backends`` and the
optional judge/vLLM modules) is torch-free, so scoring, validation and reporting run
on any machine. See ``ROADMAP.md`` for what is implemented in which milestone.
"""

__version__ = "0.0.1"
