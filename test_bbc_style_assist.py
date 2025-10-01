import json

from run_bbc_style_assist import main


def test():
    # Load LDRS golden test dataset
    input_file = "Data/golden_dataset_ldrs.jsonl"

    with open(input_file, "r", encoding="utf-8") as fp:
        samples = fp.readlines()
        samples = [json.loads(s) for s in samples]

    print(f" << * >> Loaded {len(samples)} test samples from file '{input_file}'")

    # Set hyperparameters
    decoding_methods = ["VanillaGreedy", "dola", "SLED"]
    results = []

    # Run generation for each sample
    for test_sample in samples:
        # Extract sample text
        sample_id = test_sample["example_id"]
        prompt = test_sample["prompt"]
        print(f"\n\n#################### {sample_id.upper()} ####################")

        # Run generation using three types of decoding
        sample_result = {"sample_id": sample_id, "prompt": prompt, "outputs": {}}

        for decoder in decoding_methods:
            print(f"\n-------------------- {decoder.upper()} --------------------")

            generation = main(prompt, decoder)
            sample_result["outputs"][decoder] = generation["output"]

        sample_result["sled_result_different"] = (
            sample_result["outputs"]["SLED"]
            != sample_result["outputs"]["VanillaGreedy"]
        )

        results.append(sample_result)

        print("\n-------------------- ALL RESULTS --------------------")
        results_str = json.dumps(sample_result, indent=4)
        print(results_str)

        output_file = "Results/results_golden_dataset.json"
        with open(output_file, "w", encoding="utf-8") as fp:
            fp.write(results_str)


if __name__ == "__main__":
    test()
