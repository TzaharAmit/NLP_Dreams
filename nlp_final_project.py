# -*- coding: utf-8 -*-
"""
# NLP Final Project: Task 3: Dreams
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import torch
import json
import nltk
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate import load
from prettytable import PrettyTable


"""## Dataset 1: Our Dreams, Our Selves: Automatic Interpretation of Dream Reports"""

# read data:
data = pd.read_csv('/home/linuxu/Desktop/NLP_project/rsos_dream_data.tsv', sep="\t")

# Step 1: Define Output and Interpretation Structure
# Example template for Freudian interpretation
interpretation_template = """
This dream contains significant themes:
- **Repressed Desires**: {repressed_desires}
- **Conflicts**: {conflicts}
- **Symbolic Representations**: {symbols}
- **Emotional Tone**: {emotional_tone}

- **Character Themes**: {character_summary}
- **Interaction Dynamics**: {interaction_summary}
- **Emotional Landscape**: {emotion_summary}
- **Freudian Insight**: {freudian_insight}
"""

# Step 2: Preprocessing Dataset

def clean_date(date_str):
    """Clean and normalize the dream_date field."""
    try:
        return pd.to_datetime(date_str, errors='coerce')
    except Exception as e:
        return np.nan

# Clean and normalize dream_date
data['clean_dream_date'] = data['dream_date'].apply(clean_date)
data['clean_dream_date'] = data['clean_dream_date'].fillna(pd.Timestamp('1900-01-01'))

# Tokenization and Text Cleaning for text_dream
def clean_text(text):
    """Basic text cleaning for dream text."""
    text = re.sub(r'[\n\r]', ' ', str(text))
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip().lower()

data['clean_text_dream'] = data['text_dream'].apply(clean_text)

# Normalize Numeric Features (Standard Scaling)
numeric_features = [
    'Male', 'Animal', 'Friends', 'Family', 'Dead&Imaginary',
    'Aggression/Friendliness', 'A/CIndex', 'F/CIndex', 'S/CIndex', 'NegativeEmotions'
]
scaler = StandardScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Encode dreamer and dream_language as categorical values
le_dreamer = LabelEncoder()
le_language = LabelEncoder()
data['encoded_dreamer'] = le_dreamer.fit_transform(data['dreamer'])
data['encoded_language'] = le_language.fit_transform(data['dream_language'])

# Fill missing values with defaults
for col in numeric_features:
    data[col].fillna(0, inplace=True)
data['clean_text_dream'].fillna("unknown", inplace=True)

# Step 3: Feature Engineering

def calculate_character_diversity(row):
    """Calculate character diversity based on character percentages."""
    return row[['Male', 'Animal', 'Friends', 'Family', 'Dead&Imaginary']].std()

def calculate_interaction_intensity(row):
    """Calculate overall interaction intensity based on interaction indices."""
    return (row['A/CIndex'] + row['F/CIndex'] + row['S/CIndex']) / 3

data['character_diversity'] = data.apply(calculate_character_diversity, axis=1)
data['interaction_intensity'] = data.apply(calculate_interaction_intensity, axis=1)

# Step 4: Create Synthetic Training Data

def generate_interpretation(row):
    """Generate a detailed interpretation for a given dream using HVC codes."""
    # Repressed Desires
    repressed_desires = "High" if row['NegativeEmotions'] > 0.5 else "Low"
    # Conflicts
    conflicts = "Significant" if row['Aggression/Friendliness'] > 1 else "Minimal"
    # Symbols
    symbols = "Complex" if row['character_diversity'] > 0.5 else "Simple"
    # Emotional Tone
    emotional_tone = "Negative" if row['NegativeEmotions'] > 0.5 else "Positive"
    # Character Summary
    prominent_characters = []
    if row['Male'] > 0.3:
        prominent_characters.append("male figures")
    if row['Animal'] > 0.2:
        prominent_characters.append("animals")
    if row['Family'] > 0.2:
        prominent_characters.append("family members")
    character_summary = "The dream prominently features " + ", ".join(prominent_characters) + "."
    # Interaction Summary
    interaction_summary = f"The dream contains an aggression-to-friendliness ratio of {row['Aggression/Friendliness']:.2f} and sexual interactions are {'present' if row['S/CIndex'] > 0 else 'absent'}."
    # Emotional Summary
    emotion_summary = f"The dream reflects {'strong negative emotions' if row['NegativeEmotions'] > 0.5 else 'mild emotions'}, indicating {('anxiety or conflict' if row['NegativeEmotions'] > 0.5 else 'a peaceful state')}."
    # Freudian Insight (Dynamic Logic)
    if row['Aggression/Friendliness'] > 1.5:
        if row['NegativeEmotions'] > 0.5:
            freudian_insight = "The dream suggests unresolved anger and inner conflict."
        else:
            freudian_insight = "This dream reflects dominance or assertiveness in social interactions."
    elif row['S/CIndex'] > 0.2:
        freudian_insight = "The dream highlights themes of intimacy and desire."
    elif row['character_diversity'] > 0.5:
        freudian_insight = "The dream suggests a dynamic interplay of various influences in the dreamer's life."
    elif row['Animal'] > 0.3:
        freudian_insight = "The presence of animals symbolizes primal instincts or untamed emotions."
    else:
        freudian_insight = "This dream reflects a balance of inner thoughts and social dynamics."
    row['freudian_insight'] = freudian_insight

    return interpretation_template.format(
        repressed_desires=repressed_desires,
        conflicts=conflicts,
        symbols=symbols,
        emotional_tone=emotional_tone,
        character_summary=character_summary,
        interaction_summary=interaction_summary,
        emotion_summary=emotion_summary,
        freudian_insight=freudian_insight
    )

data['synthetic_interpretation'] = data.apply(generate_interpretation, axis=1)
data['freudian_insight'] = data['synthetic_interpretation'].apply(lambda x: x.split('**Freudian Insight**: ')[1].split('\n')[0] if '**Freudian Insight**:' in x else '')

# Prepare the dataset for training
dataset = Dataset.from_pandas(data[['clean_text_dream', 'synthetic_interpretation']])
# Tokenize the data
model_name = "gpt2"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess(examples):
    return tokenizer(examples['clean_text_dream'], text_target=examples['synthetic_interpretation'], truncation=True, padding="max_length", max_length=512)
tokenized_dataset = dataset.map(preprocess, batched=True)

def preprocess(examples):
    return tokenizer(examples['clean_text_dream'], text_target=examples['synthetic_interpretation'], truncation=True, padding="max_length", max_length=512)
tokenized_dataset = dataset.map(preprocess, batched=True)
print("tokenized dataset")

# Split the dataset
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']
train_dataset.to_csv(r'./dream_llm/train_dataset.csv')
test_dataset.to_csv(r'./dream_llm/test_dataset.csv')

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name)

#Define training arguments
training_args = TrainingArguments(
    output_dir="./dream_llm",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=1,
    save_strategy="epoch",
    report_to="none"  #
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

# Fine-tune the model
trainer.train()
# Save the fine-tuned model
model.save_pretrained("./dream_llm")
tokenizer.save_pretrained("./dream_llm")

# Load model after train: in case you want to use the saved model:
# model_path = "./dream_llm"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path)
# print("Model and tokenizer loaded successfully from", model_path)

# Move model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# evaluate test dataset:
# Prepare metrics
nltk.download("punkt_tab")
bleu_scorer = nltk.translate.bleu_score
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
bert_score = load("bertscore")

# Define a perplexity function
def calculate_perplexity(predictions, tokenizer):
    """Calculate perplexity for a list of predictions."""
    log_likelihoods = []
    for pred in predictions:
        encodings = tokenizer(pred, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encodings.to('cuda'), labels=encodings["input_ids"])
        log_likelihoods.append(outputs.loss.item())
    return np.exp(np.mean(log_likelihoods))

# Generate predictions and evaluate
references = [example["synthetic_interpretation"] for example in test_dataset]
predictions = []
    
for example in test_dataset:
    input_text = example["clean_text_dream"]
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model.generate(inputs.input_ids, max_new_tokens=200, num_return_sequences=1)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions.append(pred)

# BLEU Score
bleu_scores = []
for ref, pred in zip(references, predictions):
    ref_tokens = nltk.word_tokenize(ref)
    pred_tokens = nltk.word_tokenize(pred)
    bleu_scores.append(bleu_scorer.sentence_bleu([ref_tokens], pred_tokens))
bleu_avg = np.mean(bleu_scores)
print(f"Average BLEU Score: {bleu_avg:.4f}")

# ROUGE Scores
rouge1, rouge2, rougeL = 0, 0, 0
for ref, pred in zip(references, predictions):
    scores = rouge.score(ref, pred)
    rouge1 += scores["rouge1"].fmeasure
    rouge2 += scores["rouge2"].fmeasure
    rougeL += scores["rougeL"].fmeasure
n = len(references)
print(f"ROUGE-1: {rouge1/n:.4f}, ROUGE-2: {rouge2/n:.4f}, ROUGE-L: {rougeL/n:.4f}")

# Perplexity
perplexity = calculate_perplexity(predictions, tokenizer)
print(f"Perplexity: {perplexity:.4f}")

# BERTScore
bert_score_result = bert_score.compute(predictions=predictions, references=references, lang="en")
bert_avg = np.mean(bert_score_result["f1"])
print(f"BERTScore (F1): {bert_avg:.4f}")

all_results = {"Average BLEU Score": f"{bleu_avg:.4f}",
               "ROUGE-1": f"{rouge1/n:.4f}, ROUGE-2: {rouge2/n:.4f}, ROUGE-L: {rougeL/n:.4f}",
               "Perplexity": f"{perplexity:.4f}",
               "BERTScore (F1)": f"{bert_avg:.4f}"}

filename="./dream_llm/evaluation_metrics_test.json"
with open(filename, "w") as f:
    json.dump(all_results, f, indent=4)
print(f"Evaluation metrics saved to {filename}")


#%% Model Size:
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params / (10**6)}M")
    return total_params

count_parameters(model)

#%% Validation:

"""##  List of 50 dreams and their interpretations"""

dreams_data = {
    "dream": [
        "Falling off a cliff", "Flying freely in the sky", "Teeth falling out one by one",
        "Being naked in a crowded street", "Failing an important exam", "Being chased by a shadowy figure",
        "Witnessing one's own death", "Running but moving in slow motion", "Meeting a deceased loved one",
        "Drowning in deep water", "Driving a car but losing control", "Walking through a dark forest",
        "Climbing a never-ending staircase", "Being trapped in a small room", "Losing one’s wallet or keys",
        "Arguing with a stranger", "Being late for an important event", "Watching a plane crash",
        "Standing in a burning building", "Falling into a deep abyss", "Seeing a baby crying",
        "Being unable to speak", "Walking barefoot on sharp rocks", "Eating spoiled food",
        "Discovering hidden treasure", "Losing hair suddenly", "Breaking a mirror",
        "Finding oneself in a strange house", "Witnessing a friend get hurt", "Flying but struggling to stay in the air",
        "Being in an endless maze", "Reuniting with a former lover", "Fighting with a family member",
        "Losing eyesight or going blind", "Seeing a snake in a dream", "Falling from a high building",
        "Receiving a gift from a stranger", "Being locked out of one’s home", "Discovering secret rooms in a familiar house",
        "Crossing a turbulent river", "Seeing oneself in a mirror", "Losing a beloved pet",
        "Sitting in an empty classroom", "Walking on thin ice", "Witnessing a sunrise",
        "Fighting a wild animal", "Seeing a collapsing building", "Being unable to find one’s way home",
        "Writing a letter that never gets sent", "Standing on a stage but forgetting lines"],
    "interp": [
        "Fear of losing control in life or anxieties about failure",
        "A deep desire for liberation from constraints or limitations",
        "Anxiety about physical appearance or communication breakdowns",
        "Feeling exposed or vulnerable in social or professional settings",
        "Fear of being judged or evaluated harshly by others",
        "Avoidance of unresolved fears or challenges in waking life",
        "Transition or significant changes happening in life",
        "A feeling of helplessness or frustration in achieving goals",
        "Processing grief or longing for past connections",
        "Overwhelmed by emotions or unconscious conflicts surfacing",
        "Concerns about control over life’s direction",
        "Navigating through uncertainty or the unconscious mind",
        "Struggling to achieve unattainable goals or self-improvement",
        "Feeling confined or stuck in a situation",
        "Anxiety about identity or security in waking life",
        "Internal conflicts or repressed emotions surfacing",
        "Pressure and fear of failing expectations",
        "Fears of failure or witnessing a significant loss",
        "Intense emotional stress or repressed anger",
        "Existential fears or fear of losing stability",
        "Anxiety about nurturing responsibilities or personal growth",
        "Feeling silenced or unheard in waking life",
        "Struggles or pain endured on the path to goals",
        "Guilt or discomfort with choices made recently",
        "A desire to uncover untapped potential or hidden talents",
        "Concerns about aging or loss of vitality",
        "Anxiety about self-image or fear of bad luck",
        "Exploring unknown aspects of the self",
        "Fear of losing someone or guilt over past actions",
        "Ambivalence about freedom or personal achievements",
        "Feeling lost or confused in waking life",
        "Nostalgia or unresolved emotions from past relationships",
        "Repressed anger or unresolved family tensions",
        "Fear of ignorance or losing perspective",
        "A symbol of transformation, fear, or temptation",
        "Anxiety about failure or public humiliation",
        "Expectation of unexpected opportunities or recognition",
        "Feeling disconnected from personal identity or safety",
        "Uncovering hidden aspects of the psyche",
        "Overcoming emotional obstacles or major life changes",
        "Reflecting on self-identity or inner conflicts",
        "Processing grief or emotional dependence",
        "Anxiety about learning or personal development",
        "Fear of taking risks or instability in life",
        "Hope and optimism for new beginnings",
        "Struggles with primal instincts or internal aggression",
        "Fear of losing stability or foundations in life",
        "Longing for security or emotional grounding",
        "Repressed communication or unspoken emotions",
        "Anxiety about performance or public judgment"]}
dreams_df = pd.DataFrame(dreams_data)
# Prepare the second dataset for tokenization
dream_dataset = Dataset.from_pandas(dreams_df[['dream', 'interp']])
# Tokenize the second dataset
def preprocess_dream_data(examples):
    return tokenizer(examples['dream'], text_target=examples['interp'], truncation=True, padding="max_length", max_length=512)
tokenized_dream_dataset = dream_dataset.map(preprocess_dream_data, batched=True)

# Generate predictions for the second dataset
predictions = []
references = [example["interp"] for example in tokenized_dream_dataset]

for example in tokenized_dream_dataset:
    input_text = example["dream"]
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model.generate(inputs.input_ids, max_new_tokens=200, num_return_sequences=1)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions.append(pred)

# BLEU Score
bleu_scores = []
for ref, pred in zip(references, predictions):
    ref_tokens = nltk.word_tokenize(ref)
    pred_tokens = nltk.word_tokenize(pred)
    bleu_scores.append(bleu_scorer.sentence_bleu([ref_tokens], pred_tokens))
bleu_avg = np.mean(bleu_scores)
print(f"Average BLEU Score: {bleu_avg:.4f}")

# ROUGE Scores
rouge1, rouge2, rougeL = 0, 0, 0
for ref, pred in zip(references, predictions):
    scores = rouge.score(ref, pred)
    rouge1 += scores["rouge1"].fmeasure
    rouge2 += scores["rouge2"].fmeasure
    rougeL += scores["rougeL"].fmeasure
n = len(references)
print(f"ROUGE-1: {rouge1/n:.4f}, ROUGE-2: {rouge2/n:.4f}, ROUGE-L: {rougeL/n:.4f}")

# Perplexity
perplexity = calculate_perplexity(predictions, tokenizer)
print(f"Perplexity: {perplexity:.4f}")

# BERTScore
bert_score_result = bert_score.compute(predictions=predictions, references=references, lang="en")
bert_avg = np.mean(bert_score_result["f1"])
print(f"BERTScore (F1): {bert_avg:.4f}")

all_results = {"Average BLEU Score": f"{bleu_avg:.4f}",
               "ROUGE-1": f"{rouge1/n:.4f}, ROUGE-2: {rouge2/n:.4f}, ROUGE-L: {rougeL/n:.4f}",
               "Perplexity": f"{perplexity:.4f}",
               "BERTScore (F1)": f"{bert_avg:.4f}"}

filename="./dream_llm/evaluation_metrics_val.json"
with open(filename, "w") as f:
    json.dump(all_results, f, indent=4)
print(f"Evaluation metrics saved to {filename}")
