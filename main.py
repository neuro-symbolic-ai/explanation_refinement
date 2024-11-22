import json
import argparse
import yaml
import os
from pathlib import Path
from tqdm import tqdm

from models.autoformalisation_model import AutoFormalisationModel
from models.generative_model import GPT, MistralAI, Ollama
from models.rough_inference_model import RoughInferenceModel
from models.isabelle_model import IsabelleSolver
from models.refine_model import ExplanationRefineModel


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def filter_explanations(refine_explanation,
                        unused_explanation,
                        previous_explanation):
    if not unused_explanation:
        return refine_explanation

    if not previous_explanation.strip():
        return refine_explanation

    extracted_numbers = list(set(
        [int(
            ''.join([char for char in exp if char.isdigit()])
            ) - 1 for exp in unused_explanation]
        ))
    previous_explanation_lines = [
        line.strip() for line in previous_explanation.split('\n')
        ]
    refine_explanation_lines = [
        line.strip() for line in refine_explanation.split('\n')
        ]

    for number in extracted_numbers:
        if number < len(previous_explanation_lines) and number >= 0:
            sentence_to_remove = previous_explanation_lines[number].strip()
            if sentence_to_remove in refine_explanation_lines:
                refine_explanation_lines.remove(sentence_to_remove)

    refined_explanation = '\n'.join(refine_explanation_lines)

    return refined_explanation


def main(args, config):
    model_config = config.get(args.model, {})
    engine_name = model_config.get('engine')
    api_key = model_config.get('api_key')

    # Initialise the LLM
    if 'gpt' in args.model:
        llm = GPT(engine_name, api_key)
    elif args.model == 'mistral-small':
        llm = MistralAI(engine_name, api_key)
    else:
        llm = Ollama(engine_name)

    # Load the data
    data_path = Path('./data')
    data = json.load(open(data_path / f'{args.data}.json', 'r'))

    # load the formalisation and textual inference models
    autoformalisation_model = AutoFormalisationModel(llm)
    rough_inference_model = RoughInferenceModel(llm)
    explanation_refine_model = ExplanationRefineModel(llm)

    # Run the pipeline
    for item in tqdm(data):
        premise = item['premise']
        hypothesis = item['hypothesis']
        explanation = item['explanation']
        theory_name = f'{args.data}_{item["id"]}'
        result_directory = f'result/{args.data}/{args.model}'
        os.makedirs(result_directory, exist_ok=True)

        # initialise the Isabelle solver
        isabelle_solver = IsabelleSolver(
            llm=llm,
            isabelle_session='HOL',
            port=7777,
            isabelle_name='test',
            watchdog_timeout=65,
            dirs='../../Isabelle2023'
        )
        for iteration in range(args.max_iterations):
            print(f'Iteration {iteration}')
            print(f'Premise: {premise}')
            print(f'Hypothesis: {hypothesis}')
            print(f'Explanation: {explanation}')
            print('---------------------------------')
            unused_explanation = ''
            critique_output = {}
            item[f'{iteration}it explanation'] = explanation

            # autoformalise the input sentences into Isabelle/HOL code
            isabelle_code = autoformalisation_model.formalise(
                theory_name,
                args.data,
                args.model,
                iteration,
                premise,
                explanation,
                hypothesis
            )
            # check and refine any syntax errors in the Isabelle code
            has_syntax_error = isabelle_solver.get_isabelle_syntax_output(
                isabelle_code,
                theory_name,
                args.data,
                args.model,
                iteration
            )
            if has_syntax_error:
                print(f'Iteration {iteration}: {theory_name} has syntax error')
                # contain syntax error, go next iteration
                critique_output['syntactic validity'] = has_syntax_error
                continue
            else:
                print(f'Iteration {iteration}: {theory_name} has no '
                      'syntax error')

            directory = f'formalisation/{args.data}/{args.model}/{theory_name}'
            with open(f'{directory}/{theory_name}_{iteration}.thy', 'r') as f:
                isabelle_code = f.read()
            # make the rough inference for the proof strategy
            rough_inference = rough_inference_model.get_rough_inference(
                premise,
                explanation,
                hypothesis
            )
            # print(rough_inference)

            # autoformalise the rough inference into Isabelle/HOL proof
            isabelle_code, unused_explanation = \
                autoformalisation_model.get_isabelle_proof(
                    rough_inference,
                    isabelle_code,
                    theory_name,
                    args.data,
                    args.model,
                    iteration
                )
            # print(unused_explanation)

            # check the logical validy of the constructed theory
            is_valid, error_code, inference_time = isabelle_solver.solve(
                theory_name,
                args.data,
                args.model,
                iteration,
                isabelle_code,
                explanation
            )
            print(f'logical validity of the explanation: {is_valid}')
            print(f'inference time: {inference_time}')
            print(f'proof step failed: {error_code}')
            print('---------------------------------')
            critique_output['logical validity'] = is_valid
            critique_output['syntactic validity'] = (
                True if not has_syntax_error
                else False
            )
            critique_output['proof step failed'] = error_code
            critique_output['inference time'] = inference_time

            if not is_valid and error_code == 'no':
                print('syntax error in the proof')
                print('---------------------------------')
                critique_output['syntactic validity'] = False
                item[f'{iteration}it critique output'] = critique_output
                with open(f'{result_directory}/result.json', 'w') as f:
                    json.dump(data, f, indent=4)
                continue

            if is_valid:
                if iteration != 0:
                    print('the explanation is valid and refined')
                else:
                    print('the explanation is initially valid')
                print('---------------------------------')
                item[f'{iteration}it critique output'] = critique_output
                item['refined iteration'] = iteration
                with open(f'{result_directory}/result.json', 'w') as f:
                    json.dump(data, f, indent=4)
                break
            else:
                # the explanation is logically invalid, refine the explanation
                refined_explanation = explanation_refine_model.refine(
                    premise,
                    explanation,
                    hypothesis,
                    isabelle_code,
                    error_code
                )
                # filter out unused explanation
                refined_explanation = filter_explanations(
                    refined_explanation,
                    unused_explanation,
                    explanation
                )
                explanation = refined_explanation
                item[f'{iteration}it critique output'] = critique_output
                with open(f'{result_directory}/result.json', 'w') as f:
                    json.dump(data, f, indent=4)
        isabelle_solver.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the model.')
    parser.add_argument('--model',
                        '-m',
                        type=str,
                        choices=[
                            'gpt-3.5-turbo',
                            'gpt-4o',
                            'gpt-4',
                            'mistral-small',
                            'llama2:70b',
                            'mixtral:8x7b'
                            ],
                        default='gpt-4o')
    parser.add_argument('--data',
                        '-d',
                        type=str,
                        choices=['example', 'esnli', 'qasc', 'worldtree'],
                        default='example',
                        help='The dataset to use.')
    parser.add_argument('--max_iterations',
                        '-i',
                        type=int,
                        default=11)
    config = load_config(Path('./config.yaml'))
    args = parser.parse_args()
    main(args, config)
