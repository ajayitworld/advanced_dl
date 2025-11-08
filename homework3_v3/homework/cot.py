from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        # Brief system instruction asking for concise chain-of-thought and a final answer
        system_msg = (
            "You are a helpful assistant that solves unit conversion problems. "
            "Think step-by-step (chain-of-thought) and show the short reasoning. "
            "Be concise and always finish with the answer wrapped in <answer>...</answer> tags."
        )

        # One good in-context example (question -> reasoning -> answer)
        example_q = "How many gram are there per 6 kg?"
        example_a = "Use 1 kg = 1000 grams to do the conversion: 6 * 1000 = <answer>6000</answer>"

        example_q2 = "Convert 5 quart to pint?"
        example_a2 = "Use 1 quart = 2 pints to do the conversion: 5 * 2 = <answer>10</answer>"

        example_q3 = "What is the measurement of 3 kg when converted into pounds?"
        example_a3 = "Use 1 kg = 2.20462 pounds to do the conversion: 3 * 2.20462 = <answer>6.61386</answer>"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": example_q},
            {"role": "assistant", "content": example_a},
            {"role": "user", "content": example_q2},
            {"role": "assistant", "content": example_a2},
            {"role": "user", "content": example_q3},
            {"role": "assistant", "content": example_a3},
            {"role": "user", "content": question},
        ]

        # Convert the chat messages into the model's chat template string.
        # add_generation_prompt=True will append the assistant generation prompt token
        # tokenize=False returns the raw string (not token ids)
        formatted_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        # Print the formatted prompt to see the chat template structure
        #print("\nFormatted Chat Template:")
        #print("="*50)
        #print(formatted_prompt)
        #print("="*50)
        
        return formatted_prompt


def load(checkpoint: str | None = None) -> CoTModel:
    return CoTModel(checkpoint)


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
