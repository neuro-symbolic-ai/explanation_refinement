class ExplanationRefineModel:
    def __init__(self, llm):
        self.llm = llm

    def refine(self,
               premise: str,
               explanation: str,
               hypothesis: str,
               isabelle_code: str,
               error_code: str
               ) -> str:
        if premise == 'none':
            inference_result = self.llm.generate(
                model_prompt_dir='textual_inference',
                prompt_name='refine_no_premise_prompt',
                isablle_code=isabelle_code,
                proof_step=error_code,
                explanation=explanation,
                hypothesis=hypothesis,
                numbered_list=True,
                remove_number=True
            )
        else:
            inference_result = self.llm.generate(
                model_prompt_dir='textual_inference',
                prompt_name='refine_with_premise_prompt',
                isabelle_code=isabelle_code,
                proof_step=error_code,
                premise=premise,
                explanation=explanation,
                hypothesis=hypothesis,
                numbered_list=True,
                remove_number=True
            )
        return inference_result
