import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.translate.bleu_score import sentence_bleu

# Load model directly
model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./models_cache")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./models_cache")

# Ensure tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def load_and_preprocess_data(train_file, test_file):
    """ Load and preprocess the dataset """
    print("Loading and preprocessing data...")

    # Check if files exist
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError("Train or Test CSV file not found.")

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Ensure required columns exist
    required_columns = {"question", "answer"}
    if not required_columns.issubset(train_data.columns) or not required_columns.issubset(test_data.columns):
        raise ValueError(f"Missing required columns in CSV. Expected: {required_columns}")

    # Preprocess train and test data
    train_samples = [{"question": row["question"], "answer": row["answer"]} for _, row in train_data.iterrows()]
    test_samples = [{"question": row["question"], "answer": row["answer"]} for _, row in test_data.iterrows()]

    print("Data loading and preprocessing completed.")
    return train_samples, test_samples


def fine_tune_opt_model(train_samples):
    """ Fine-tune OPT model on the training dataset """
    print("Starting fine-tuning of the OPT model...")

    # Define the custom dataset class
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

            # Tokenize inputs and targets
            inputs = self.tokenizer(
                input_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            targets = self.tokenizer(
                target_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Prepare labels by ignoring padding tokens
            labels = targets["input_ids"].squeeze()
            labels[labels == self.tokenizer.pad_token_id] = -100

            return {
                "input_ids": inputs["input_ids"].squeeze(),
                "attention_mask": inputs["attention_mask"].squeeze(),
                "labels": labels,
            }

    # Tokenize input and target texts
    input_texts = [f"Question: {sample['question']} Answer:" for sample in train_samples]
    target_texts = [sample["answer"] for sample in train_samples]

    # Prepare dataset
    train_dataset = CustomDataset(input_texts, target_texts, tokenizer)

    # Fine-tuning arguments
    training_args = TrainingArguments(
        output_dir="./opt_task_expert",
        evaluation_strategy="no",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,  # Improved batching
    )

    trainer.train()
    print("Fine-tuning completed.")

    # Save the fine-tuned model
    model.save_pretrained("./opt_task_expert")
    tokenizer.save_pretrained("./opt_task_expert")
    print("Fine-tuned model saved.")


def evaluate_opt_model(test_samples):
    """ Evaluate the fine-tuned model """
    print("Starting evaluation of the fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained("./opt_task_expert")
    model = AutoModelForCausalLM.from_pretrained("./opt_task_expert")

    results = []
    for sample in test_samples:
        input_text = f"Question: {sample['question']} Answer:"
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50)
        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(input_text, "").strip()

        results.append({"question": sample["question"], "generated_answer": generated_answer})

    print("Evaluation completed.")
    return results


# Main execution
if __name__ == "__main__":
    train_file = "dataset/clean_main_train.csv"
    test_file = "dataset/clean_main_test.csv"

    # Load and preprocess data
    train_samples, test_samples = load_and_preprocess_data(train_file, test_file)

    # Fine-tune the OPT model
    fine_tune_opt_model(train_samples)

    # Evaluate the fine-tuned model
    results = evaluate_opt_model(test_samples)
    results_df = pd.DataFrame(results)

    # Save results in dateOUT directory
    output_dir = "dateOUT"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "evaluation_results.csv")

    results_df.to_csv(output_file, index=False)
    print(f"Results saved successfully to {output_file}.")

    # Compute BLEU scores for accuracy measurement
    results_df["bleu_score"] = results_df.apply(
        lambda row: sentence_bleu([row["question"].split()], row["generated_answer"].split())
        if isinstance(row["question"], str) and isinstance(row["generated_answer"], str) else 0,
        axis=1
    )

    # Compute and print average BLEU score
    average_bleu_score = results_df["bleu_score"].mean()
    print(f"Average BLEU Score: {average_bleu_score:.4f}")

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

        # BLEU Score Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(results_df["bleu_score"], bins=20, alpha=0.7, color="blue")
        plt.title("BLEU Score Distribution")
        plt.xlabel("BLEU Score")
        plt.ylabel("Frequency")
        plt.show()
    else:
        print("No data available for visualizations.")
