from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Define model path
model_path = os.path.join("final-model-merged")

# Load model and tokenizer
try:
    logging.info("Loading tokenizer and model from: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    logging.info("Model successfully loaded on %s.", device)

except Exception as e:
    logging.error("Error loading model: %s", e)
    raise SystemExit("Failed to load model. Exiting...")

# Flask endpoint
@app.route("/ask", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"error": "No question provided"}), 400

        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=256,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id
            )

        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        return jsonify({"answer": answer})

    except Exception as e:
        logging.error("Error during generation: %s", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
