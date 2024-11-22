class RoughInferenceModel:
    def __init__(self, llm):
        self.llm = llm

    def get_rough_inference(self, premise: str, explanation: str,
                            hypothesis: str) -> str:
        inference_result = self.llm.generate(
            model_prompt_dir='textual_inference',
            prompt_name='get_rough_inference_prompt',
            premise=premise,
            explanation=explanation,
            hypothesis=hypothesis
        )
        return inference_result
