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
    evolution_rate: float,
    evolution_scale: int,
    repetition_penalty: float,
) -> dict:
    # Hyperparameters
    model_name = "meta-llama/Llama-2-7b-hf"
    num_gpus = "auto"
    max_gpu_memory = 80
    device = "cuda"
    early_exit_layers = None

    max_new_tokens = 256
    temperature = 0.9
    relative_top = 0.1
    relative_top_value = -1000.0

    do_sample = False
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
    completion_response, c_dist = model.generate(
        prompt,
        do_sample=do_sample,
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
        "c_dist": c_dist,
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
    parser.add_argument("--evolution_rate", type=float, default=2)
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

    # "You are a BBC Journalist. Please find below a news article from the Local Democracy Reporting Service. Please re-write the article in the style of a BBC News Article? Do not change any of the facts in the article - simply modify the text to follow the BBC's style guide.\n\nLDRS Article:\nEast Riding leisure centres could start closing earlier, the council portfolio holder in charge of them has said.\nEast Riding Council's Culture and Leisure Portfolio Holder Cllr Mike Medini said there were no plans to close any leisure centres but some could shut earlier when visitor numbers are low.\nThe portfolio holder added the authority was doing everything it could to lessen the impact on leisure services after councils elsewhere had cut them back.\nA full meeting of East Riding Council also saw Council Leader Cllr Jonathan Owen say some serious decisions around spending lay ahead amid mounting financial uncertainty.\nThe leader said senior councillors were currently in talks with officers about its budget position going forward amid a widening hole in the authority's finances.\nHe added rising energy costs and high pay settlements were among the factors driving up spending as inflation continues to climb and the cost of living crisis deepens.\nCllr Medini said leisure centres' important role in helping communities mattered more than ever given the current climate.\nBut he added the council was looking at scaling opening hours back.\nThe portfolio holder said: \"I'd like to reassure residents, visitors and businesses that there are no plans to close leisure centres.\n\"But we are looking at closing some pools a few minutes early late at night when there's very little demand.\n\"Our approach is to regularly review and we're doing everything we can to reduce the impact on centres.\n\"Leisure centres play a vital role in supporting communities, they help in the council's prevention and early intervention approach to health and they help improve quality of life.\n\"There's never been a more important time to support communities.\"\n\nBBC Article:\n"
    main(prompt, decoding_method, evolution_rate, evolution_scale, repetition_penalty)
