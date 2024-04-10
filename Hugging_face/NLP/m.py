# %%
# Transformers installation
#! pip install transformers datasets
# To install from source instead of the last release, comment the command above and uncomment the following one.
# ! pip install git+https://github.com/huggingface/transformers.git

# %% [markdown]
# # Fine-tune a pretrained model

# %% [markdown]
# There are significant benefits to using a pretrained model. It reduces computation costs, your carbon footprint, and allows you to use state-of-the-art models without having to train one from scratch. 🤗 Transformers provides access to thousands of pretrained models for a wide range of tasks. When you use a pretrained model, you train it on a dataset specific to your task. This is known as fine-tuning, an incredibly powerful training technique. In this tutorial, you will fine-tune a pretrained model with a deep learning framework of your choice:
# 
# * Fine-tune a pretrained model with 🤗 Transformers [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer).
# * Fine-tune a pretrained model in TensorFlow with Keras.
# * Fine-tune a pretrained model in native PyTorch.
# 
# <a id='data-processing'></a>

# %% [markdown]
# ## Prepare a dataset

# %% [markdown]
# Before you can fine-tune a pretrained model, download a dataset and prepare it for training. The previous tutorial showed you how to process data for training, and now you get an opportunity to put those skills to the test!
# 
# Begin by loading the [Yelp Reviews](https://huggingface.co/datasets/yelp_review_full) dataset:

# %%
from datasets import load_dataset

dataset = load_dataset("yelp_review_full")
dataset["train"][100]

# %% [markdown]
# As you now know, you need a tokenizer to process the text and include a padding and truncation strategy to handle any variable sequence lengths. To process your dataset in one step, use 🤗 Datasets [`map`](https://huggingface.co/docs/datasets/process.html#map) method to apply a preprocessing function over the entire dataset:

# %%
dataset

# %%
dataset['train'].features

# %%
# # pip install accelerate
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")

# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)


# tokenized_datasets = dataset.map(tokenize_function, batched=True)

# %%
from transformers import AutoTokenizer

checkpoint = "bert-base-cased"


tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# %% [markdown]
# If you like, you can create a smaller subset of the full dataset to fine-tune on to reduce the time it takes:

# %%
tokenized_datasets['train']

# %%
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10000))

# %% [markdown]
# <a id='trainer'></a>

# %%
small_train_dataset

# %% [markdown]
# ## Train

# %% [markdown]
# At this point, you should follow the section corresponding to the framework you want to use. You can use the links
# in the right sidebar to jump to the one you want - and if you want to hide all of the content for a given framework,
# just use the button at the top-right of that framework's block!

# %% [markdown]
# ## Train with PyTorch Trainer

# %% [markdown]
# 🤗 Transformers provides a [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) class optimized for training 🤗 Transformers models, making it easier to start training without manually writing your own training loop. The [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) API supports a wide range of training options and features such as logging, gradient accumulation, and mixed precision.
# 
# Start by loading your model and specify the number of expected labels. From the Yelp Review [dataset card](https://huggingface.co/datasets/yelp_review_full#data-fields), you know there are five labels:

# %%
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=5)

# %% [markdown]
# <Tip>
# 
# You will see a warning about some of the pretrained weights not being used and some weights being randomly
# initialized. Don't worry, this is completely normal! The pretrained head of the BERT model is discarded, and replaced with a randomly initialized classification head. You will fine-tune this new model head on your sequence classification task, transferring the knowledge of the pretrained model to it.
# 
# </Tip>

# %% [markdown]
# ### Training hyperparameters

# %% [markdown]
# Next, create a [TrainingArguments](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments) class which contains all the hyperparameters you can tune as well as flags for activating different training options. For this tutorial you can start with the default training [hyperparameters](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments), but feel free to experiment with these to find your optimal settings.
# 
# Specify where to save the checkpoints from your training:

# %% [markdown]
# ### Evaluate

# %% [markdown]
# [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) does not automatically evaluate model performance during training. You'll need to pass [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) a function to compute and report metrics. The [🤗 Evaluate](https://huggingface.co/docs/evaluate/index) library provides a simple [`accuracy`](https://huggingface.co/spaces/evaluate-metric/accuracy) function you can load with the [evaluate.load](https://huggingface.co/docs/evaluate/main/en/package_reference/loading_methods#evaluate.load) (see this [quicktour](https://huggingface.co/docs/evaluate/a_quick_tour) for more information) function:

# %%
import numpy as np
import evaluate

metric = evaluate.load("accuracy")

# %% [markdown]
# Call `compute` on `metric` to calculate the accuracy of your predictions. Before passing your predictions to `compute`, you need to convert the predictions to logits (remember all 🤗 Transformers models return logits):

# %%
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# %% [markdown]
# ### Trainer

# %% [markdown]
# Create a [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) object with your model, training arguments, training and test datasets, and evaluation function:

# %%
import time
import torch
from transformers import TrainerCallback, TrainingArguments, Trainer,TrainerCallback

class GPUMonitoringCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated()
            gpu_memory_reserved = torch.cuda.memory_reserved()
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_usage = (gpu_memory_allocated + gpu_memory_reserved) / gpu_memory_total * 100
            gpu_memory_usage_gb = (gpu_memory_allocated + gpu_memory_reserved) / (1024 * 1024 * 1024)  # Convert to GB
            
            elapsed_time = time.time() - self.start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            print(f"\rStep: {state.global_step}, GPU Memory Usage: {gpu_memory_usage_gb:.2f} GB ({gpu_memory_usage:.2f}%), Elapsed Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}", end="", flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTotal Training Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

# Create an instance of the GPUMonitoringCallback
gpu_monitoring_callback = GPUMonitoringCallback()




training_args = TrainingArguments(
    output_dir="my_awesome_food_model1",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=90,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    
   # push_to_hub=True,
)


# Create an instance of the Trainer with the callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[gpu_monitoring_callback],
)

torch.cuda.empty_cache()

# Train the model
trainer.train()

# %% [markdown]
# 

# %%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

# %% [markdown]
# Then fine-tune your model by calling [train()](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer.train):

# %% [markdown]
# <a id='pytorch_native'></a>

# %%
del model
del trainer
torch.cuda.empty_cache()


