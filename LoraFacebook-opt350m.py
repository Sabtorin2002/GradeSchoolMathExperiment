import os
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig, TaskType
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.translate.bleu_score import sentence_bleu

# Load model and tokenizer
model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./models_cache")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./models_cache")


def load_and_preprocess_data(train_file, test_file):
    """Load and preprocess the dataset"""
    print("Loading and preprocessing data...")
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    train_samples = [{"question": row["question"], "answer": row["answer"]} for _, row in train_data.iterrows()]
    test_samples = [{"question": row["question"], "answer": row["answer"]} for _, row in test_data.iterrows()]

    print("Data loading and preprocessing completed.")
    return train_samples, test_samples


def fine_tune_opt_model(train_samples):
    """Fine-tune OPT model using LoRA for efficient training"""
    print("Starting LoRA fine-tuning of the OPT model...")

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, input_texts, target_texts, tokenizer, max_length=512):
            self.input_texts = input_texts
            self.target_texts = target_texts
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.input_texts)

        def __getitem__(self, idx):
            input_text = self.input_texts[idx]
            target_text = self.target_texts[idx]

            inputs = self.tokenizer(
                input_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            targets = self.tokenizer(
                target_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
            )

            labels = targets["input_ids"].squeeze()
            labels[labels == self.tokenizer.pad_token_id] = -100

            return {
                "input_ids": inputs["input_ids"].squeeze(),
                "attention_mask": inputs["attention_mask"].squeeze(),
                "labels": labels,
            }

    input_texts = [f"Question: {sample['question']} Answer:" for sample in train_samples]
    target_texts = [sample["answer"] for sample in train_samples]
    train_dataset = CustomDataset(input_texts, target_texts, tokenizer)

    # LoRA Configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Rank of LoRA matrices
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],  # Apply LoRA to key attention projection layers
    )

    model_lora = get_peft_model(model, lora_config)
    model_lora.print_trainable_parameters()  # Print trainable parameters (LoRA)

    training_args = TrainingArguments(
        output_dir="./opt_task_expert_lora",
        evaluation_strategy="no",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model_lora,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda data: {
            "input_ids": torch.stack([f["input_ids"] for f in data]),
            "attention_mask": torch.stack([f["attention_mask"] for f in data]),
            "labels": torch.stack([f["labels"] for f in data]),
        },
    )

    trainer.train()
    print("LoRA fine-tuning completed.")

    # Save LoRA adapter (instead of full model)
    model_lora.save_pretrained("./opt_task_expert_lora")
    tokenizer.save_pretrained("./opt_task_expert_lora")
    print("LoRA fine-tuned model saved.")


def evaluate_opt_model(test_samples):
    """Evaluate the fine-tuned model"""
    print("Starting evaluation of the fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained("./opt_task_expert_lora")
    model = AutoModelForCausalLM.from_pretrained("./opt_task_expert_lora")

    results = []
    for sample in test_samples:
        input_text = f"Question: {sample['question']} Answer:"
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50)
        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({"question": sample["question"], "generated_answer": generated_answer})

    print("Evaluation completed.")
    return results


results = []
results_df = pd.DataFrame()  # Ensure results_df is defined

if __name__ == "__main__":
    train_file = "dataset/clean_main_test.csv"
    test_file = "dataset/clean_main_test.csv"

    train_samples, test_samples = load_and_preprocess_data(train_file, test_file)

    # Fine-tune the OPT model with LoRA
    fine_tune_opt_model(train_samples)

    # Evaluate the fine-tuned model
    try:
        results = evaluate_opt_model(test_samples)
        results_df = pd.DataFrame(results)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        results = []

# Save results
output_dir = "dateOUT"
os.makedirs(output_dir, exist_ok=True)

if not results_df.empty:
    output_file = os.path.join(output_dir, "evaluation_results.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Results saved successfully to {output_file}.")
else:
    print("No results to save. Evaluation might have failed.")

# Visualizations
if not results_df.empty:
    # Word Cloud
    text = " ".join(results_df["generated_answer"].dropna().apply(str))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Generated Answers")
    plt.show()

    # BLEU Scores
    results_df["bleu_score"] = results_df.apply(
        lambda row: sentence_bleu([row["question"].split()], row["generated_answer"].split())
        if row["question"] and row["generated_answer"] else 0,
        axis=1
    )
    plt.figure(figsize=(10, 6))
    plt.hist(results_df["bleu_score"], bins=20, alpha=0.7, color="blue")
    plt.title("BLEU Score Distribution")
    plt.xlabel("BLEU Score")
    plt.ylabel("Frequency")
    plt.show()
else:
    print("No data available for visualizations.")
