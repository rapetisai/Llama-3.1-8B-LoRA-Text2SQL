# Llama-3.1-8B-LoRA-Text2SQL 🚀

This project fine-tunes **Meta-Llama-3.1-8B** using **LoRA (Low-Rank Adaptation)** and **4-bit quantization** for the **Text-to-SQL task**.  
The model is trained on the [`gretelai/synthetic_text_to_sql`](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) dataset, which provides natural language instructions paired with SQL queries.

---

## 📖 What is this project about?

Natural language interfaces to databases are becoming essential in analytics, business intelligence, and software systems. Instead of writing SQL manually, users can describe queries in plain English — e.g.:


This model learns to **translate instructions into SQL queries**, making database interaction more accessible to non-technical users.

---

## ⚡ Key Features
- **Base Model:** [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama) (via [Unsloth](https://github.com/unslothai/unsloth) for efficiency).  
- **Parameter-Efficient Training:** Uses **LoRA/PEFT** to fine-tune only a fraction of the model weights → faster & cheaper training.  
- **Memory Efficient:** 4-bit quantization with `bitsandbytes` → fits large models on consumer-grade GPUs (like Colab).  
- **Dataset:** `gretelai/synthetic_text_to_sql` (natural language + SQL pairs).  
- **Training Framework:** Hugging Face `transformers`, `trl`, and `datasets`.

---

## ⚙️ Training Configuration
- `per_device_train_batch_size=4`  
- `gradient_accumulation_steps=1`  
- `warmup_steps=100`  
- `max_steps=1000`  
- `learning_rate=2e-5`  
- `weight_decay=0.01`  
- Mixed precision (`fp16`/`bf16` based on GPU)  

---

## 🔧 Workflow
1. Install dependencies (transformers, unsloth, bitsandbytes, trl).  
2. Load the base model in **4-bit quantization**.  
3. Inject **LoRA adapters** for efficient fine-tuning.  
4. Load and preprocess the **Text-to-SQL dataset**.  
5. Fine-tune the model with Hugging Face `Trainer`.  
6. Save the trained model and tokenizer.  
7. Run inference to translate new instructions into SQL queries.

---

## 🎯 Why is this useful?
- **Bridges the gap** between non-technical users and databases.  
- **Cheaper & faster** than full fine-tuning thanks to LoRA + quantization.  
- Provides a **reproducible pipeline** for instruction tuning LLMs on structured tasks.  
- Can be extended to other domains (e.g., **NL → Python code**, **NL → API calls**).

---

## 📉 Training Loss

The curve below is from the fine-tuning run (max_steps=1000, lr=2e-5, batch_size=4).

## 🔮 Future Work

Extend training to real-world Text-to-SQL datasets (Spider, WikiSQL).

Add evaluation metrics such as exact match and execution accuracy.

Optimize LoRA configurations (different r, dropout, and layer targeting).

Deploy as an interactive app using Gradio/Streamlit for non-technical users.

Explore cross-domain adaptation (natural language → Python code, API queries, etc.).
