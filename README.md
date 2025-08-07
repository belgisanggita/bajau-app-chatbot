# Bajau App Chatbot Agent

This project demonstrates fine-tuning the open-source TinyLLaMA model to build a chatbot agent capable of handling instruction-style and procedural questions. As a case study, I use the “Bajau E-commerce App” scenario to illustrate how the agent responds to real-world queries.

---

## Model Result
| Metric         |  Score |
| -------------- | -----: |
| **ROUGE-1**    | 0.9367 |
| **ROUGE-2**    | 0.8703 |
| **ROUGE-L**    | 0.9089 |
| **ROUGE-LSum** | 0.9343 |
| **BLEU**       | 0.8782 |

---

## Open Source Model Selection

**Chosen model:** TinyLLaMA
**Rationale:**

* **Lightweight & Efficient**: Smaller footprint makes training and deployment feasible on limited compute resources.
* **Fast Turnaround**: Reduced parameter count accelerates fine-tuning iterations.
* **Sufficient Capacity**: Delivers strong results on procedural text despite its compact size.

---

## Data Preparation & Preprocessing

1. **Synthetic Data Generation**

   * I used the LLaMA-3 base model to automatically generate a diverse set of instruction-answer pairs.

2. **Cleaning & Formatting**

   * Ensured data was clean, valid JSON, and properly structured.

3. **Tokenization**

   * Applied the TinyLLaMA tokenizer to convert raw text into model inputs.

4. **Dataset Structure & Split**

   ```json
   {
     "input":  "<instruction or question>",
     "output": "<model response>"
   }
   ```

   * **Training set**: 80%
   * **Test set**: 20%

---

## Fine-Tuning Strategy

* **Framework**: PEFT (Parameter-Efficient Fine-Tuning) using **LoRA** (Low Rank Adaptation) adapters.
* **Key Hyperparameters**:

  * LoRA rank, alpha, and dropout
  * Batch size
  * Number of epochs
  * Learning rate

> **Challenge:** Resource constraints limited to modest batch sizes and epochs. Despite this, the model achieved strong performance.

---

## Evaluation & Benchmarking

I measured text generation quality using ROUGE and BLEU scores.

**BLEU Details**

* Precisions (1-gram to 4-gram): 94.24%, 90.16%, 85.13%, 82.25%
* Brevity Penalty: 1.0
* Length ratio (generated/reference): 1.0006 (1562/1561)

Beyond numeric metrics, I conducted qualitative tests on:

* **Seen questions** (from the training set)
* **Unseen, real-world questions** (new cases outside training/test data)

---

## Installation & Usage

Download the model using this link: https://drive.google.com/file/d/1_9wCICmM8tUPe_wq-O8GUQC33IX7nHLy/view?usp=drive_link
```bash
# Clone the repository
git clone https://github.com/belgisanggita/bajau-app-chatbot.git
cd bajau-app-chatbot

# Install dependencies
pip install -r requirements.txt

# Start the API server
python app.py
```
## 📁 Directory Structure of the APP BAJAU project.

```

bajau-app-chatbot/
├── README.md        
└── src/
    └── final-model-merged/     
    └── app.py                    
    └── dataset.json              
    └── finetuning.ipynb
    └── requirements.txt 

````

Once the server is running, you can send POST requests to the `/ask` endpoint.
For example, using [Postman](https://www.postman.com/) or `curl`:

```bash
curl -X POST http://localhost:5000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "How can I swap a bundle order for a different bundle in the Bajau E-Commerce mobile app?"}'
```
<img width="1919" height="1021" alt="Image" src="https://github.com/user-attachments/assets/046620d8-8563-48a3-ae2f-61ea5cd845cf" />


