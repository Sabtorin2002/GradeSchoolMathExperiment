# Grade School Math Experiment

This project explores the use of language models, such as Facebook OPT-125M and LoRa, to solve Grade School Math (GSM) problems. The objective is to evaluate the performance of these models in understanding and solving basic math problems using fine-tuning techniques.

## Table of Contents
- [Introduction](#introduction)
- [Models Used](#models-used)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)

## Introduction
The Grade School Math (GSM) task requires models to solve basic math problems using natural language understanding. This project implements and evaluates the performance of the following models:

1. Facebook OPT-125M
2. Facebook OPT-350M
3. LoRa fine-tuned versions of these models

The goal is to compare the baseline performance of the models against their fine-tuned counterparts.

## Models Used
### Facebook OPT-125M
A small-scale version of the OPT (Open Pretrained Transformer) series, designed for efficient NLP tasks.

### Facebook OPT-350M
A larger version of the OPT series, offering improved performance on a variety of tasks due to its increased model size.

### LoRa (Low-Rank Adaptation)
LoRa is a fine-tuning technique used to efficiently adapt pre-trained models to specific tasks. In this project, LoRa was applied to both OPT-125M and OPT-350M for GSM tasks.

## Dataset
The dataset used for this project consists of grade school math problems formatted as text. Key datasets used include:
- Cleaned training and test sets, such as `clean_main_train.csv` and `facebook125_small_dataset.csv`.
- The dataset contains addition, subtraction, multiplication, and division problems.

## Methodology
1. **Baseline Testing**: Evaluate the performance of the original OPT models on the GSM dataset.
2. **Fine-Tuning**: Use LoRa to fine-tune the OPT models on the GSM dataset.
3. **Evaluation**: Compare the performance of the baseline and fine-tuned models using metrics such as accuracy and loss.

## Results
- The LoRa fine-tuned models demonstrated improved performance over the baseline models in terms of accuracy.
- Detailed results can be found in the `results` directory or in the associated report.

## Requirements
To run this project, install the following dependencies:

```bash
pip install torch transformers datasets
```

## Usage
### Training
To fine-tune the models, run the following scripts:

- For OPT-125M:
  ```bash
  python LoraFacebook-opt125m.py
  ```

- For OPT-350M:
  ```bash
  python LoraFacebook-opt350m.py
  ```

### Evaluation
Evaluate the fine-tuned models using the provided test scripts:

```bash
python opt_task_expert_lora.py
```

### Results
The results will be stored in the `results` directory.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

