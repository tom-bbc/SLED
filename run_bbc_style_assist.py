import json
import warnings

import transformers
from sled_decoding import SLED_DecodedLLM_GSM8K as SLED_DecodedLLM
from utils.utils_gsm8k import build_prompt, clean_answer, is_correct, set_seed

transformers.logging.set_verbosity(100)


def main():
    # Hyperparameters
    model_name = "meta-llama/Llama-2-7b-hf"
    num_gpus = "auto"
    max_gpu_memory = 80
    device = "cpu"
    data_path = "Data/FACTOR/wiki_factor.csv"
    output_path = "./gsm8k_result"
    early_exit_layers = None

    max_new_tokens = 256
    top_p = 0.95
    top_k = 0
    temperature = 0.9
    repetition_penalty = 1.0
    relative_top = 0.1
    relative_top_value = -1000.0

    do_sample = True
    do_shuffle = False
    seed = 42
    decoding_method = "VanillaGreedy"  # Choices: "VanillaGreedy","SLED", "dola"
    evolution_rate = 2
    evolution_scale = 10

    set_seed(seed)

    # Instantiate LLM with SLED decoding
    model = SLED_DecodedLLM(model_name, device, num_gpus, max_gpu_memory)
    stop_word_list = ["Q:", "\\end{code}", "\n", ". "]
    model.set_stop_words(stop_word_list)
    print(f"Instantiated model: '{model_name}'")

    # Setup layers to use in decoding process
    if decoding_method == "VanillaGreedy":
        if early_exit_layers is not None:
            warnings.warn(
                "The 'early_exit_layers' argument should be None when using Vanilla greedy decoding."
            )

        mature_layer = None
        candidate_premature_layers = None
        print("Vanilla greedy decoding from the final layer")

    else:
        if early_exit_layers is None:
            early_exit_layers = [int(x) for x in range(model.num_layers + 1)]
        else:
            early_exit_layers = [int(x) for x in early_exit_layers.split(",")]

        mature_layer = early_exit_layers[-1]
        candidate_premature_layers = early_exit_layers[:-1]

        print(
            f"MODE: {decoding_method} decoding with the final layer: {mature_layer} and premature layers: {candidate_premature_layers}"  # noqa
        )

    # Formulate model input structures
    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
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

    input_text = "Q: Jack has a stack of books that is 12 inches thick. He knows from experience that 80 pages is one inch thick. If he has 6 books, how many pages is each one on average? A:"  # noqa
    label = "There are 960 pages because 80 x 12 = <<80*12=960>>960 Each book is 160 pages because 960 / 6 = <<960/6=160>>160 #### 160"  # noqa

    # input_text = build_prompt(sample, do_shuffle)
    print(f"Formatted prompt: '{input_text}'")

    # Generate response over data sample
    model_completion, c_dist = model.generate(input_text, **generate_kwargs)
    print(f"Generated completion: {model_completion}")

    model_answer = clean_answer(model_completion)
    print(f"Cleaned response: {model_answer}")

    # Formulate model output structures
    result_dict = {}
    is_cor = is_correct(model_answer, label)

    result_dict["is_correct"] = is_cor
    result_dict["model_answer"] = model_answer
    result_dict["model_completion"] = model_completion
    result_dict["full_input_text"] = input_text

    print("Results:")
    print(json.dumps(result_dict, indent=4))

    return result_dict


if __name__ == "__main__":
    main()
