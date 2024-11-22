import re
import os


class AutoFormalisationModel:
    def __init__(self, llm):
        self.llm = llm
        self.code = ''

    def _add_quotes(self, isabelle_code: str) -> str:
        assumes_pattern = r'(assumes asm: )(.*)'
        shows_pattern = r'(shows )(.*)'

        def add_quotes_to_line(match):
            content = match.group(2)
            if not content.startswith('"') and not content.endswith('"'):
                content = f'"{content}"'
            return f'{match.group(1)}{content}'
        isabelle_code = re.sub(assumes_pattern,
                               add_quotes_to_line, isabelle_code)
        isabelle_code = re.sub(shows_pattern,
                               add_quotes_to_line, isabelle_code)
        return isabelle_code

    def _fix_assume_quantifier(self, isabelle_code: str) -> str:
        def replace_quantifier(match):
            quantifier_str = match.group(1)
            new_quantifier_str = re.sub(r'[∀∃].*?\.\s', '', quantifier_str)
            return f'assumes asm: "{new_quantifier_str}"'

        assumes_pattern = r'assumes asm: "(.*?)"'
        isabelle_code = re.sub(assumes_pattern, replace_quantifier,
                               isabelle_code)

        return isabelle_code

    def _clean_proof(self, isabelle_code: str) -> str:
        pattern = r'(proof -).*?(qed)(?!.*qed)'
        return re.sub(pattern, r'\1  \n  \n  \n\2',
                      isabelle_code, flags=re.DOTALL)

    def _remove_brackets(self, isabelle_code: str) -> str:
        assumes_pattern = r'(assumes asm: ")(.+)(")'
        shows_pattern = r'(shows ")(.+)(")'
        assumes_match = re.search(assumes_pattern, isabelle_code)
        if assumes_match:
            assumes_content = assumes_match.group(2)
            if '(' in assumes_content and ')' in assumes_content:
                assumes_content = re.sub(r'[\(\),]', ' ', assumes_content)
                isabelle_code = isabelle_code[:assumes_match.start(2)] + \
                    assumes_content + isabelle_code[assumes_match.end(2):]
        shows_match = re.search(shows_pattern, isabelle_code)
        if shows_match:
            shows_content = shows_match.group(2)
            if '(' in shows_content and ')' in shows_content:
                shows_content = re.sub(r'[\(\),]', ' ', shows_content)
                isabelle_code = isabelle_code[:shows_match.start(2)] + \
                    shows_content + isabelle_code[shows_match.end(2):]
        return isabelle_code

    def _get_event(self, premise: str,
                   explanation: str,
                   hypothesis: str) -> str:
        def format_sentences(sentences, type):
            formatted = f"{type}:\n"
            sentence_list = sentences.split('\n') if '\n' in sentences \
                else [sentences]
            for order, sentence in enumerate(sentence_list, start=1):
                if sentence.strip():
                    formatted += (f"{order}. {sentence}\n"
                                  "Has Action:\nActions:\n\n")
            return formatted

        nl_knowledge_base = (
            format_sentences(hypothesis, "Hypothesis Sentence") +
            format_sentences(explanation, "Explanation Sentence") +
            format_sentences(premise, "Premise Sentence")
        )

        inference_result = self.llm.generate(
            model_prompt_dir='autoformalisation',
            prompt_name='get_event_prompt',
            input_sentence=nl_knowledge_base
        )
        inference_result = re.sub(r'^.*?answer:\s*', '', inference_result,
                                  flags=re.DOTALL | re.IGNORECASE)
        if 'premise sentence' not in inference_result.lower():
            inference_result += ('\nPremise Sentence:\n1. none\n'
                                 'Has Action: No\nActions:')
        last_action_index = inference_result.lower().rfind('actions:')
        if last_action_index != -1:
            action_end_index = inference_result.find('\n', last_action_index)
            if action_end_index != -1:
                inference_result = inference_result[:action_end_index].strip()
        return inference_result

    def _get_davidsonian_form(self, input_sentence: str) -> str:

        inference_result = self.llm.generate(
            model_prompt_dir='autoformalisation',
            prompt_name='get_davidsonian_form_prompt',
            input_sentence=input_sentence
        )

        inference_result = '\n'.join([
            line for line in inference_result.split('\n')
            if line and (line.startswith(("Hypothesis", "Explanation",
                                          "Premise", "Logical")) or
                         re.match(r'^\d+\.', line))
        ])

        last_action_index = inference_result.lower().rfind('logical form:')
        if last_action_index != -1:
            action_end_index = inference_result.find('\n', last_action_index)
            if action_end_index != -1:
                inference_result = inference_result[:action_end_index].strip()
        lines = inference_result.split('\n')
        cleaned_lines = [line for line in lines
                         if "Provided sentences" not in line]
        for i in range(len(cleaned_lines) - 1, 0, -1):
            if cleaned_lines[i-1].startswith("Logical"):
                cleaned_lines.insert(i, "")

        cleaned_inference_result = '\n'.join(cleaned_lines)

        if 'premise sentence' not in cleaned_inference_result.lower():
            cleaned_inference_result += ('\n\nPremise Sentence:\n'
                                         'Logical form: none')

        return cleaned_inference_result

    def _get_axioms(self, davidsonian_form: str) -> str:
        lower_case_result = davidsonian_form.lower()
        start_index = lower_case_result.rfind("explanation sentence")
        if start_index != -1:
            substring_after_explanation = lower_case_result[start_index:]
            end_index_relative = substring_after_explanation.rfind(
                "premise sentence"
            )
            if end_index_relative != -1:
                end_index = start_index + end_index_relative
            else:
                end_index = len(davidsonian_form)
        else:
            end_index = -1
        if start_index != -1 and end_index != -1:
            explanatory_sentences = \
                davidsonian_form[start_index:end_index].strip()
        else:
            explanatory_sentences = ""

        inference_result = self.llm.generate(
            model_prompt_dir='autoformalisation',
            prompt_name='get_isabelle_axiom_prompt',
            explanatory_sentences=explanatory_sentences
        )

        inference_result = ("imports Main\n\n" +
                            inference_result +
                            "\ntheorem hypothesis:\n assumes asm: \n" +
                            " shows \nproof -\n  \n  \nqed\n\nend")

        return inference_result

    def _get_theorem(self, davidsonian_form: str, axiom: str,
                     premise: str) -> str:
        hypothesis_index = davidsonian_form.lower().rfind("hypothesis")
        if hypothesis_index != -1:
            davidsonian_form = davidsonian_form[hypothesis_index:]
        if premise != 'none':
            explanation_pattern = (
                r"(?:Explanation Sentence|Explanation Sentences|"
                r"Explanation sentence|Explanation sentences):"
                r".*?"
                r"(?=Premise Sentence|Premise Sentences|"
                r"Premise sentence|Premise sentences)"
            )
            input_sentence = re.sub(explanation_pattern, '',
                                    davidsonian_form,
                                    flags=re.DOTALL | re.IGNORECASE)
        else:
            explanation_pattern = (
                r"(?:Explanation Sentence|Explanation Sentences|"
                r"Explanation sentence|Explanation sentences).*"
            )
            input_sentence = re.sub(explanation_pattern, '',
                                    davidsonian_form,
                                    flags=re.DOTALL | re.IGNORECASE)

        if premise != 'none':
            prompt_file = 'get_isabelle_theorem_with_premise_prompt'
        else:
            prompt_file = 'get_isabelle_theorem_no_premise_prompt'

        for _ in range(5):
            inference_result = self.llm.generate(
                model_prompt_dir='autoformalisation',
                prompt_name=prompt_file,
                input_sentence=input_sentence,
                axiom_code=axiom
            )

            if "qed" in inference_result and "end" in inference_result:
                inference_result = self._add_quotes(inference_result)
                inference_result = self._fix_assume_quantifier(
                    inference_result
                )
                # inference_result = self._remove_brackets(inference_result)
                return inference_result
        return inference_result

    def _get_unused_explanations(self) -> list[str]:
        isabelle_code = self.code
        proof_pattern = r'proof -.*?qed'
        match = re.search(proof_pattern, isabelle_code, re.DOTALL)
        if match:
            proof_code = match.group(0)
        else:
            proof_code = ''
        explanation_matches = re.findall(r'explanation_\d+', proof_code)
        used_explanations = '\n'.join(explanation_matches)
        collected_explanations = used_explanations.split('\n')
        all_explanations = set(re.findall(r"(explanation_\d+)", isabelle_code))
        unused_explanations = [exp for exp in all_explanations if
                               exp not in collected_explanations]

        return unused_explanations

    def get_isabelle_proof(self, rough_inference: str,
                           isabelle_code: str,
                           theory_name: str,
                           data_name: str,
                           model_name: str,
                           iteration: int) -> str:
        isabelle_code = self._clean_proof(isabelle_code)
        unused_explanation = ''
        for _ in range(5):
            inference_result = self.llm.generate(
                model_prompt_dir='autoformalisation',
                prompt_name='get_isabelle_proof_prompt',
                isabelle_code=isabelle_code,
                rough_inference=rough_inference,
            )
            if 'proof -' in inference_result and 'qed' in inference_result:
                pattern = r"```(.*?)```"
                if "```" in inference_result:
                    match = re.search(pattern, inference_result, re.DOTALL)
                    if match:
                        inference_result = match.group(1)
                proof_content_pattern = r'proof -.*?qed'
                match = re.search(proof_content_pattern, inference_result,
                                  re.DOTALL)
                if match:
                    inference_result = match.group(0)
                proof_pattern = r'proof -.*?qed'
                isabelle_code = re.sub(proof_pattern, inference_result,
                                       isabelle_code, flags=re.DOTALL)
                isabelle_code = self._fix_assume_quantifier(isabelle_code)
                self.save_formalised_kb(
                    isabelle_code,
                    theory_name,
                    data_name,
                    model_name,
                    iteration
                )
                unused_explanation = self._get_unused_explanations()
                return isabelle_code, unused_explanation
        return isabelle_code, unused_explanation

    def fix_inner_syntax_error(self, isabelle_code: str,
                               error_detail: str, inner_code: str) -> str:
        refined_code = self.llm.generate(
            model_prompt_dir='autoformalisation',
            prompt_name='fix_inner_syntax_error_prompt',
            code=isabelle_code,
            error_detail=error_detail,
            code_cause_error=inner_code
        )
        refined_code = self._clean_proof(refined_code)
        refined_code = self._add_quotes(refined_code)
        refined_code = self._fix_assume_quantifier(refined_code)
        return refined_code

    def fix_contradiction_error(self, isabelle_code: str,
                                contradiction_code: str) -> str:
        refined_code = self.llm.generate(
            model_prompt_dir='autoformalisation',
            prompt_name='fix_contradiction_error_prompt',
            natural_language=self.logical_form,
            code=isabelle_code,
            code_cause_error=contradiction_code
        )
        refined_code = self._clean_proof(refined_code)
        refined_code = self._add_quotes(refined_code)
        refined_code = self._fix_assume_quantifier(refined_code)
        return refined_code

    def formalise(self, theory_name: str, data_name: str, model_name: str,
                  iteration: int, premise: str, explanation: str,
                  hypothesis: str) -> str:
        event_semantics = self._get_event(premise, explanation, hypothesis)
        davidsonian_form = self._get_davidsonian_form(event_semantics)
        axioms = self._get_axioms(davidsonian_form)
        theorem = self._get_theorem(davidsonian_form, axioms, premise)
        self.code = f'theory {theory_name}\n' + theorem
        self.code = self._clean_proof(self.code)
        self.save_formalised_kb(self.code, theory_name,
                                data_name, model_name,
                                iteration)
        return self.code

    def save_formalised_kb(self,
                           isabelle_code: str,
                           theory_name: str,
                           data_name: str,
                           model_name: str,
                           iteration: int) -> None:
        isabelle_code = re.sub(r'.*imports Main', 'imports Main',
                               isabelle_code, flags=re.DOTALL)
        isabelle_code = f'theory {theory_name}_{iteration}\n' + isabelle_code
        self.code = isabelle_code
        directory = f'formalisation/{data_name}/{model_name}/{theory_name}'
        os.makedirs(directory, exist_ok=True)
        with open(f'{directory}/{theory_name}_{iteration}.thy', 'w') as f:
            f.write(isabelle_code)
