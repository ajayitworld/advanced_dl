from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input to SmolLM2. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        You don't need to change this function for now.
        """
        return question

    def parse_answer(self, answer: str) -> float:
        """
        Parse the <answer></answer> tag and return a float.
        This function is somewhat robust to output errors (e.g. missing </answer> tags).
        """
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str) -> str:
        """
        (Optional) Implement this method first and then implement batched_generate below.
        It is much easier to implement generation without batching.

        The overall flow is the same:
        - tokenize the prompt with self.tokenizer
        - call self.model.generate
        - decode the outputs with self.tokenizer.decode

        """
        return self.batched_generate([prompt])[0]

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        """
        Batched version of `generate` method.
        This version returns a single generation for each prompt.
        """

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        """
        Batched version of `generate` method.

        You will likely get an up to 10x speedup using batched decoding.

        To implement batch decoding you will need to:
        - tokenize the prompts self.tokenizer with padding=True and return_tensors="pt"
        - call self.model.generate
        - decode the outputs with self.tokenizer.batch_decode

        Tip: You need to set self.tokenizer.padding_side = "left" to get the correct padding behavior for generation.
             Left padding makes sure all sequences are aligned to the right (i.e. where tokens are generated).
        Tip: self.model.generate takes a lot of parameters. Here are some relevant ones:
            - max_new_tokens: The maximum number of tokens to generate. Set this to a reasonable value
                              (50 should suffice).
            - do_sample and temperature: For any temperature > 0, set do_sample=True.
                                         do_sample=False will use greedy decoding.
            - num_return_sequences: The number of sequences to return. Note that this will generate a flat
                                    list of len(prompts) * num_return_sequences entries.
            - eos_token_id: The end of sequence token id. This is used to stop generation. Set this
                            to self.tokenizer.eos_token_id.
        Pro Tip: Only batch_decode generated tokens by masking out the inputs with
                 outputs[:, len(inputs["input_ids"][0]) :]
        """
        from tqdm import tqdm  # Importing tqdm for progress bar

        # Preventing OOM
        # Depending on your GPU batched generation will use a lot of memory.
        # If you run out of memory, try to reduce the micro_batch_size.
        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            return [
                r
                for idx in tqdm(
                    range(0, len(prompts), micro_batch_size), desc=f"LLM Running on Micro Batches {micro_batch_size}"
                )
                for r in self.batched_generate(prompts[idx : idx + micro_batch_size], num_return_sequences, temperature)
            ]

        # Quick return for empty input
        if len(prompts) == 0:
            return [] if num_return_sequences is None else []

        # Ensure left padding so generation aligns on the right
        self.tokenizer.padding_side = "left"

        # Tokenize prompts with padding (align on right) and get attention mask
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Configure generation parameters
        do_sample = temperature is not None and temperature > 0
        n_return = 1 if num_return_sequences is None else int(num_return_sequences)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            do_sample=do_sample,
            temperature=float(temperature),
            num_return_sequences=n_return,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # outputs shape: (batch_size * num_return_sequences, seq_len)
        #input_len = input_ids.shape[1]

        # Only decode the newly generated tokens (mask out the prompt)
        #gen_tokens = outputs[:, input_len:]
        gen_tokens = outputs[:, len(inputs["input_ids"][0]) :]
        # Move to CPU and convert to python lists for tokenizer decoding
        gen_tokens_list = gen_tokens.cpu().tolist()

        # Decode generated tokens
        decoded = self.tokenizer.batch_decode(gen_tokens_list, skip_special_tokens=True)

        # If single return sequence per prompt, return flat list[str]
        if num_return_sequences is None:
            # decoded is already in order corresponding to prompts
            return decoded

        # Otherwise, group decoded outputs per prompt
        grouped: list[list[str]] = []
        for i in range(0, len(decoded), n_return):
            grouped.append(decoded[i : i + n_return])

        return grouped
    

        '''
        outputs_cpu = outputs.cpu().tolist()
        decoded_full = self.tokenizer.batch_decode(outputs_cpu, skip_special_tokens=True)

        # Reconstruct prompt text per sample (without padding) using attention mask
        # inputs["input_ids"] is padded on the left; attention_mask indicates real tokens.
        input_ids_cpu = inputs["input_ids"].tolist()
        attention_mask_cpu = inputs.get("attention_mask")
        if attention_mask_cpu is not None:
            attention_mask_cpu = attention_mask_cpu.tolist()
        else:
            # If no attention mask provided, assume all tokens are real
            attention_mask_cpu = [[1] * len(ids) for ids in input_ids_cpu]

        prompt_texts: list[str] = []
        for ids, mask in zip(input_ids_cpu, attention_mask_cpu):
            # extract non-padded token ids
            prompt_ids = [tok for tok, m in zip(ids, mask) if m]
            prompt_texts.append(self.tokenizer.decode(prompt_ids, skip_special_tokens=True))

        # Now strip the prompt_text from the decoded_full to get continuations
        continuations: list[str] = []
        for i_out, full in enumerate(decoded_full):
            # Determine which prompt this output corresponds to
            prompt_idx = i_out // n_return
            prompt_txt = prompt_texts[prompt_idx]
            # If full starts with prompt_txt, strip it; otherwise try a safer rsplit
            if full.startswith(prompt_txt):
                cont = full[len(prompt_txt) :]
            else:
                # fallback: remove the first occurrence of prompt_txt if present
                idx = full.find(prompt_txt)
                if idx != -1:
                    cont = full[idx + len(prompt_txt) :]
                else:
                    # last resort: assume full is just the continuation
                    cont = full

            continuations.append(cont)

        # If single return sequence per prompt, return flat list[str]
        if num_return_sequences is None:
            return continuations

        # Otherwise, group decoded outputs per prompt
        grouped: list[list[str]] = []
        for i in range(0, len(continuations), n_return):
            grouped.append(continuations[i : i + n_return])

        return grouped
        '''

    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        # Convert each question
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        return [self.parse_answer(g) for g in generations]


def test_model():
    # The following code simply tests of the BaseLLM is able to complete text.
    # It should produce garbage answers, but it should not crash.
    # In my case it talks about cats eating cats, and dogs being happy.
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
