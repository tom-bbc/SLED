import argparse
import json
import warnings

import transformers
from sled_decoding import SLED_DecodedLLM_GSM8K as SLED_DecodedLLM
from utils.utils_gsm8k import build_prompt, clean_answer, is_correct, set_seed

transformers.logging.set_verbosity(100)


def main(prompt: str, decoding_method: str) -> dict:
    # Hyperparameters
    model_name = "meta-llama/Llama-2-7b-hf"
    num_gpus = "auto"
    max_gpu_memory = 80
    device = "cuda"
    early_exit_layers = None

    max_new_tokens = 256
    temperature = 0.9
    repetition_penalty = 1.0
    relative_top = 0.1
    relative_top_value = -1000.0

    seed = 42
    evolution_rate = 2
    evolution_scale = 10

    set_seed(seed)

    # Instantiate LLM with SLED decoding
    model = SLED_DecodedLLM(model_name, device, num_gpus, max_gpu_memory)
    stop_word_list = ["Q:", "\\end{code}", "\n", ". "]
    model.set_stop_words(stop_word_list)
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
    completion_response, c_dist = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
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
    if completion_response.startswith("\n"):
        cleaned_response = completion_response[1:]
        cleaned_response = cleaned_response.split["\n"][0]
    else:
        cleaned_response = completion_response.split["\n"][0]

    # Formulate model output structures
    result_dict = {
        "prompt": prompt,
        "raw_completion": completion_response,
        "c_dist": c_dist,
    }

    print(" << * >> Results:")
    print(json.dumps(result_dict, indent=4))

    return result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", "-p", type=str, default=None)
    parser.add_argument(
        "--decoding-method",
        "-d",
        type=str,
        default="VanillaGreedy",
        choices=["VanillaGreedy", "SLED", "dola"],
    )

    args = parser.parse_args()
    prompt = args.prompt
    decoding_method = args.decoding_method

    if prompt is None:
        prompt = "Jack has a stack of books that is 12 inches thick. He knows from experience that 80 pages is one inch thick. If he has 6 books, how many pages is each one on average?"  # noqa

    main(prompt, decoding_method)
