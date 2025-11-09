from .base_llm import BaseLLM
from .sft import test_model, data_collator, tokenize

# used copilot, chatgpt and other online help
def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name
    print('model_path ', model_path)

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str = "homework/rft_model",
    **kwargs,
):
    # Reuse much of the SFT code here
    from pathlib import Path
    import json
    import torch
    from peft import get_peft_model, LoraConfig, TaskType
    from transformers import TrainingArguments, Trainer

    # Load the RFT dataset
    rft_path = Path(kwargs.get("rft_json", "data/rft.json"))
    #print('rft_path ', rft_path)
    if not rft_path.exists():
        print(f"RFT dataset not found at {rft_path}. Run datagen.generate_dataset first.")
        print("  python -m homework.datagen")                
        return

    with rft_path.open() as f:
        entries = json.load(f)

    # entries are [question, answer, reasoning]
    #print('entries ', entries)

    llm = BaseLLM()

    # Configure LoRA
    lora_r = int(kwargs.get("lora_r", 16))
    lora_alpha = int(kwargs.get("lora_alpha", 64))
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
    )

    model = get_peft_model(llm.model, lora_config)
    model.train()
    if torch.cuda.is_available():
        model.enable_input_require_grads()

    def format_rft_example(question: str, answer: str, reasoning: str) -> dict[str,str]:
      return {
        "question": question,
        "answer": reasoning  # Reasoning includes the full chain of thought + answer
      }

    # Build a simple TensorDataset-like mapping for Trainer
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, tokenizer, toks, format_fn):
            self.toks = toks
            self.format_fn = format_fn
            self.tokenizer  = tokenizer

        def __len__(self):
            return len(self.toks) #["input_ids"].shape[0]

        def __getitem__(self, idx):
            #return {k: v[idx] for k, v in self.toks.items()}             
             formatted_data = self.format_fn(*self.toks[idx])
             #print('formatted_data', formatted_data)
             return tokenize(self.tokenizer, **formatted_data)


    train_ds = SimpleDataset(llm.tokenizer, entries, format_rft_example)

    # Training arguments
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print('output_path ', output_path)
    

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=int(kwargs.get("num_epochs", 3)),
        per_device_train_batch_size=int(kwargs.get("batch_size", 16)),
        learning_rate=float(kwargs.get("learning_rate", 1e-3)),
        logging_dir=str(output_path),
        logging_steps=10,
        save_strategy="epoch",
        report_to="tensorboard",
        gradient_checkpointing=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )
    
    trainer.train()

    # Save LoRA adapter
    print(f"Saving model to {output_path}...")
    trainer.save_model(Path(output_path))

    print("RFT training finished and saved to", str(output_path))

    # Test the model
    print("\nTesting trained model...")
    test_model(output_path)


def test_model(ckpt_path: str = "homework/rft_model"):
    """Test the RFT model"""
    from .data import Dataset, benchmark
    from peft import PeftModel
    
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
    llm.model.eval()

    benchmark_result = benchmark(llm, testset, 100)
    print('benchmark_result', benchmark_result)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")

if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
