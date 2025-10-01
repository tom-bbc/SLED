import argparse
import json
import re
import warnings

import transformers
from sled_decoding import SLED_DecodedLLM_GSM8K as SLED_DecodedLLM
from utils.utils_gsm8k import set_seed

transformers.logging.set_verbosity(100)


def main(
    prompt: str,
    decoding_method: str,
    evolution_rate: int = 2,
    evolution_scale: int = 10,
    repetition_penalty: float = 1.0,
) -> dict:
    # Hyperparameters
    # model_name = "meta-llama/Llama-2-7b-hf"
    model_name = "meta-llama/Llama-3.1-8B"

    device = "cuda"
    num_gpus = "auto"
    max_gpu_memory = 80

    early_exit_layers = None
    relative_top = 0.1
    relative_top_value = -1000.0
    do_sample = False
    max_new_tokens = 1024
    temperature = 1.0
    top_p = 1.0

    seed = 42
    set_seed(seed)

    # Instantiate LLM with SLED decoding
    model = SLED_DecodedLLM(model_name, device, num_gpus, max_gpu_memory)

    # stop_word_list = ["Q:", "\\end{code}", "\n", ". "]
    # model.set_stop_words(stop_word_list)

    print(f"\n << * >> Instantiated model: '{model_name}'")

    # Setup layers to use in decoding process
    if decoding_method == "VanillaGreedy":
        if early_exit_layers is not None:
            warnings.warn(
                " -- * -- The 'early_exit_layers' argument should be None when using Vanilla greedy decoding."
            )

        mature_layer = None
        candidate_premature_layers = None
        print(" << * >> Decoding mode: Vanilla greedy decoding from the final layer")

    else:
        if early_exit_layers is None:
            early_exit_layers = [int(x) for x in range(model.num_layers + 1)]
        else:
            early_exit_layers = [int(x) for x in early_exit_layers.split(",")]

        mature_layer = early_exit_layers[-1]
        candidate_premature_layers = early_exit_layers[:-1]

        print(
            f" << * >> Decoding mode: {decoding_method} decoding"
            f"\n << * >> Final layer: {mature_layer}"
            f"\n << * >> Premature layers: {candidate_premature_layers}"
        )

    # Formulate model input structures
    print(f" << * >> Input prompt: \n{prompt}", end="\n\n")

    # Generate response over data sample
    completion_response, _ = model.generate(
        prompt,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        mode=decoding_method,
        mature_layer=mature_layer,
        candidate_premature_layers=candidate_premature_layers,
        relative_top=relative_top,
        relative_top_value=relative_top_value,
        evolution_rate=evolution_rate,
        evolution_scale=evolution_scale,
    )

    print(f" << * >> Generated completion: \n{completion_response}", end="\n\n")

    # Clean generated output
    cleaned_response = re.sub(r"\\n", " ", completion_response)

    cleaned_response = cleaned_response.strip()
    prompt = prompt.strip()

    if cleaned_response.startswith(prompt):
        cleaned_response = cleaned_response[len(prompt) :]
        cleaned_response = cleaned_response.strip()

    # Formulate model output structures
    result_dict = {
        "prompt": prompt,
        "raw_completion": completion_response,
        "output": cleaned_response,
    }

    print(" << * >> Results:")
    print(json.dumps(result_dict, indent=4))

    return result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", "-p", type=str, default=None)
    parser.add_argument(
        "--decoding_method",
        "-d",
        type=str,
        default="VanillaGreedy",
        choices=["VanillaGreedy", "SLED", "dola"],
    )
    parser.add_argument("--evolution_rate", type=int, default=2)
    parser.add_argument("--evolution_scale", type=int, default=10)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    args = parser.parse_args()
    prompt = args.prompt
    decoding_method = args.decoding_method
    evolution_rate = args.evolution_rate
    evolution_scale = args.evolution_scale
    repetition_penalty = args.repetition_penalty

    if prompt is None:
        prompt = "Jack has a stack of books that is 12 inches thick. He knows from experience that 80 pages is one inch thick. If he has 6 books, how many pages is each one on average?"  # noqa

    main(prompt, decoding_method, evolution_rate, evolution_scale, repetition_penalty)
