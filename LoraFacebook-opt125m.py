import os
import re
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator)
from peft import LoraConfig, get_peft_model, TaskType
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu

# Ensure stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", cache_dir="./models_cache")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", cache_dir="./models_cache")

# Ensure tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# LoRA configuration
lora_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=16,
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


def preprocess_text(text):
    """Preprocess text by lowercasing, removing punctuation, and stopwords."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text


def load_and_preprocess_data(train_file, test_file):
    """ Load and preprocess the dataset """
    print("Loading and preprocessing data...")
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError("Train or Test CSV file not found.")

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    required_columns = {"question", "answer"}
    if not required_columns.issubset(train_data.columns) or not required_columns.issubset(test_data.columns):
        raise ValueError(f"Missing required columns in CSV. Expected: {required_columns}")

    train_samples = [{"question": preprocess_text(row["question"]), "answer": preprocess_text(row["answer"])}
                     for _, row in train_data.iterrows()]
    test_samples = [{"question": preprocess_text(row["question"]), "answer": preprocess_text(row["answer"])}
                    for _, row in test_data.iterrows()]

    print("Data loading and preprocessing completed.")
    return train_samples, test_samples


def fine_tune_lora_model(train_samples):
    """ Fine-tune LoRA-optimized OPT model """
    print("Starting fine-tuning...")

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
            inputs = self.tokenizer(input_text, max_length=self.max_length, padding="max_length", truncation=True,
                                    return_tensors="pt")
            targets = self.tokenizer(target_text, max_length=self.max_length, padding="max_length", truncation=True,
                                     return_tensors="pt")
            labels = targets["input_ids"].squeeze()
            labels[labels == tokenizer.pad_token_id] = -100
            return {"input_ids": inputs["input_ids"].squeeze(), "attention_mask": inputs["attention_mask"].squeeze(),
                    "labels": labels}

    input_texts = [f"Question: {sample['question']} Answer:" for sample in train_samples]
    target_texts = [sample["answer"] for sample in train_samples]
    train_dataset = CustomDataset(input_texts, target_texts, tokenizer)

    training_args = TrainingArguments(
        output_dir="./lora_opt_expert",
        evaluation_strategy="no",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()
    print("Fine-tuning completed.")
    model.save_pretrained("./lora_opt_expert")
    tokenizer.save_pretrained("./lora_opt_expert")
    print("Fine-tuned model saved.")


def evaluate_lora_model(test_samples):
    """ Evaluate the fine-tuned LoRA model """
    print("Starting evaluation...")
    tokenizer = AutoTokenizer.from_pretrained("./lora_opt_expert")
    model = AutoModelForCausalLM.from_pretrained("./lora_opt_expert")

    results = []
    for sample in test_samples:
        input_text = f"Question: {sample['question']} Answer:"
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50)
        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(input_text, "").strip()
        results.append({"question": sample["question"], "generated_answer": generated_answer})

    print("Evaluation completed.")
    return results


if __name__ == "__main__":
    train_file = "dataset/clean_main_train.csv"
    test_file = "dataset/clean_main_test.csv"
    train_samples, test_samples = load_and_preprocess_data(train_file, test_file)
    fine_tune_lora_model(train_samples)
    results = evaluate_lora_model(test_samples)
    results_df = pd.DataFrame(results)
    results_df.to_csv("dateOUT/evaluation_results.csv", index=False)

    results_df["bleu_score"] = results_df.apply(
        lambda row: sentence_bleu([row["question"].split()], row["generated_answer"].split())
        if isinstance(row["question"], str) and isinstance(row["generated_answer"], str) else 0,
        axis=1
    )
    print(f"Average BLEU Score: {results_df['bleu_score'].mean():.4f}")
