<img width="696" height="314" alt="image" src="https://github.com/user-attachments/assets/5927f53d-2563-4f58-ba10-28f43e2a920f" /># TinyLlama Fine-Tuning on Marine Biology Domain with LoRA

This project demonstrates how to adapt the TinyLlama-1.1B-Chat model to a specialized domain (marine biology) using Parameter-Efficient Fine-Tuning (PEFT) with LoRA and 4-bit quantization. It showcases the effectiveness of instruction-style fine-tuning in low-resource environments and highlights the capability of pretrained LLMs to generalize and specialize simultaneously.

## Project Overview

- **Model**: [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- **Domain**: Marine Biology Q&A
- **Fine-Tuning Method**: LoRA + 4-bit Quantization using Hugging Face PEFT and bitsandbytes
- **Instruction Style**: `<|user|>` and `<|assistant|>` formatted Q&A turns
- **Compute Requirements**: Runs on a single consumer GPU (Google Colab NVIDIA Tesla T4 (16 GB VRAM))

## Running the Notebook

This project is fully contained in a single Jupyter notebook:

- `MarineBio-TinyLlama-FineTuning-LoRA.ipynb`

The notebook is executable in a single pass and includes:
- Environment setup
- Dataset preparation
- Fine-tuning loop using PEFT
- Evaluation using multiple metrics (ROUGE, BERTScore, Token F1, TF-IDF similarity)
- Visualization of performance improvements

## Highlights

- Fine-tuned TinyLlama-1.1B-Chat using Parameter-Efficient Fine-Tuning with LoRA and 4-bit quantization to adapt the model to the marine biology domain with minimal compute and memory overhead.
- Only 1.13% of model parameters (~12.6M) were updated, enabling low-resource fine-tuning on consumer GPUs while preserving the base model's general capabilities.
- A new instruction-style Q&A dataset was created for the marine biology domain, addressing a previously unmet need for such domain-specific instruction tuning data.
- Highlights the potential of pretrained LLMs to assist in generating high-quality synthetic datasets in low-resource domains—enabling bootstrapping of instruction-tuning tasks even when real-world labeled data is scarce.
- Instruction fine-tuning performed using a small synthetic dataset (~150 examples) formatted with `<|user|>` and `<|assistant|>` tokens, reflecting real-world dialogue structure.
- Quantitative improvements were observed across lexical overlap, semantic similarity, and answer span matching metrics after fine-tuning. The fine-tuned model produced more focused, specific answers, while the base model gave broader, generic responses—highlighting the powerful prior knowledge embedded in pretrained LLMs and the effectiveness of targeted fine-tuning.
- Emergent generalization: The fine-tuned model exhibited better use of marine biology terminology and structure, despite the low-data setting, while the base model, despite no prior exposure to domain-specific examples, generated broader answers—demonstrating the strong general knowledge embedded in pretrained LLMs.
- Demonstrates that pretrained LLMs already encode broad, transferable knowledge. Using LoRA, they can be adapted efficiently to domain-specific tasks, validating the strength of transfer learning in modern NLP.


## References

- Dataset: [Marine Biology Q&A Dataset](https://huggingface.co/datasets/enigma04/marine-biology-qna-dataset)
- TinyLlama Model: [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- Hugging Face PEFT Library: [PEFT Documentation](https://huggingface.co/docs/peft/index)
- LoRA Paper: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Parameter-Efficient Fine-Tuning Blog: [Hugging Face PEFT Blog](https://huggingface.co/blog/peft)
- Evaluation Frameworks: [Evaluating Large Language Models](https://arxiv.org/abs/2307.03109)
- Gemini API References:
  - [GenerationConfig](https://ai.google.dev/api/generate-content#v1beta.GenerationConfig)
  - [Text Generation Documentation](https://ai.google.dev/gemini-api/docs/text-generation)


