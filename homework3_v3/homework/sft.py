from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: float) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    Round to 3 decimal places for cleaner training targets while maintaining accuracy.
    """
    # Format answer to 3 decimal places to avoid float precision issues
    answer_str = f"<answer>{answer:.3f}</answer>"
    return {
        "question": prompt,
        "answer": answer_str
    }


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def get_lora_param_size(model) -> float:
    """Calculate size of LoRA parameters in MB"""
    import numpy as np
    lora_params = 0
    for n, p in model.named_parameters():
        if 'lora_' in n:
            lora_params += np.prod(p.shape)
    return (lora_params * 4) / (1024 * 1024)  # Size in MB assuming float32

def train_model(
    output_dir: str,
    num_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 2e-4,
    lora_r: int = 8,  # rank of LoRA adapters
    lora_alpha: int = 32,  # alpha scaling, typically 4-8x rank
    **kwargs,
):
    """Train the model using LoRA adapters and HuggingFace Trainer.
    
    Args:
        output_dir: Directory to save checkpoints and logs
        num_epochs: Number of training epochs (default 5)
        batch_size: Batch size per device (default 32)
        learning_rate: Learning rate (default 2e-4)
        lora_r: LoRA rank dimension (default 8)
        lora_alpha: LoRA alpha scaling factor (default 32)
    """
    from pathlib import Path
    from peft import get_peft_model, LoraConfig, TaskType
    from transformers import TrainingArguments, Trainer
    import torch

    # Initialize base model and dataset
    llm = BaseLLM()
    trainset = Dataset("train")
    
    # Configure LoRA - use all linear layers, no bias
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        bias="none",
    )
    
    # Convert to LoRA model
    model = get_peft_model(llm.model, lora_config)
    if torch.cuda.is_available():
        model.enable_input_require_grads()  # Required for gradient checkpointing + LoRA
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        logging_dir=str(output_dir),
        logging_steps=10,
        save_strategy="epoch",
        report_to="tensorboard",
        gradient_checkpointing=True,
        # Add fp16 for faster training if supported
        fp16=torch.cuda.is_available(),
        # Optimize memory usage
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
    )
    
    # Create dataset with our formatting
    dataset = TokenizedDataset(llm.tokenizer, trainset, format_example)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda x: x,  # Identity since dataset already returns proper format
    )
    
    # Train and save
    trainer.train()
    
    # Save the final model to sft_model directory for the grader
    final_output = Path(__file__).parent / "sft_model"
    model.save_pretrained(final_output)
    
    # Run test to verify
    test_model(final_output)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
