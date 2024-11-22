from abc import ABC, abstractmethod
from typing import Optional
import re
import tenacity

from mistralai import Mistral
from openai import OpenAI
import ollama

from models.prompt_model import PromptModel


class GenerativeModel(ABC):
    def __init__(self, model_name,
                 prompt_model=None):
        self.model_name = model_name
        self.prompt_model = prompt_model

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass

    def extract_code(self, result):
        pattern = r"```(.*?)```"
        match = re.search(pattern, result, re.DOTALL)
        if match:
            result = match.group(1)
        return result

    def extract_numbered_list(self, model_reponse: str,
                              prefix: Optional[str] = None,
                              remove_number: Optional[bool] = False) -> str:
        """
        Extracts a numbered list from the given result string.

        Parameters:
        - model_reponse (str): The input string from which to
          extract the numbered list.
        - prefix (str, optional): A prefix to match before the
          numbered list. If provided, only lists following this
          prefix will be extracted.
        - remove_number (bool, optional): If True, the extracted list
          will be cleaned of numbering.
        """
        if prefix:
            pattern = re.compile(
                rf'{re.escape(prefix)}\s*((?:\d+\..*?(?:\n|$))+)',
                re.IGNORECASE | re.DOTALL)
            match = pattern.search(model_reponse)
            if match:
                extracted = match.group(1)
            else:
                extracted = model_reponse
        else:
            numbered_list_pattern = re.compile(r'(\d+\..*?(?:\n|$))+',
                                               re.IGNORECASE | re.DOTALL)
            match = numbered_list_pattern.search(model_reponse)
            extracted = match.group(0) if match else model_reponse

        if remove_number:
            cleaned = re.sub(r'\d+\.\s*', '', extracted)
            return '\n'.join(cleaned.splitlines()).strip()
        else:
            return extracted


class GPT(GenerativeModel):
    def __init__(self, model_name, api_key, prompt_model=None):
        super().__init__(model_name, prompt_model)
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        if prompt_model is None:
            self.prompt_model = PromptModel()

    # handle rate limit
    @tenacity.retry(wait=tenacity.wait_exponential(
            multiplier=1, min=4, max=30))
    def completion_with_backoff(self, **kwargs):
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as e:
            print(f'Error: {e}')
            raise e

    def generate(self,
                 model_prompt_dir: str,
                 prompt_name: str,
                 prefix: Optional[str] = None,
                 numbered_list: Optional[bool] = False,
                 remove_number: Optional[bool] = False,
                 test: Optional[bool] = False,
                 **replacements) -> str:
        """
        Generates a response from the LLM model.

        Parameters:
        - model_prompt_dir (str): The directory of the model prompt.
        - prompt_name (str): The name of the prompt.
        - prefix (str, optional): A prefix to match before the numbered list.
        - numbered_list (bool, optional): If True, the response will be
          extracted as a numbered list.
        - remove_number (bool, optional): If True, the numbered list will
          be cleaned of numbering.
        """
        system_prompt, user_prompt = self.prompt_model.process_prompt(
            model_name=model_prompt_dir,
            prompt_name=prompt_name,
            **replacements
        )

        response = None
        try:
            response = self.completion_with_backoff(
                model=self.model_name,
                temperature=0,
                frequency_penalty=0,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True
            )
        except Exception as e:
            print(f'Error: {e}')
            return
        result = []
        for chunk in response:
            if hasattr(chunk, 'choices') and len(
                chunk.choices) > 0 and hasattr(
                    chunk.choices[0], 'delta') and chunk.choices[
                        0].delta.content is not None:
                result.append(str(chunk.choices[0].delta.content))
        result = ''.join(result)
        if test:
            print(result)
        # post processing
        if "```" in result and not numbered_list:
            # extract code from code blocks
            result = self.extract_code(result)
        elif numbered_list and (prefix is not None
                                or re.search(r'\d+\.', result)):
            result = self.extract_numbered_list(result, prefix,
                                                remove_number)
        return result


class MistralAI(GenerativeModel):
    def __init__(self, model_name, api_key, prompt_model=None):
        super().__init__(model_name, prompt_model)
        self.api_key = api_key
        self.client = Mistral(api_key=self.api_key)
        if prompt_model is None:
            self.prompt_model = PromptModel()

    def generate(self,
                 model_prompt_dir: str,
                 prompt_name: str,
                 prefix: Optional[str] = None,
                 numbered_list: Optional[bool] = False,
                 remove_number: Optional[bool] = False,
                 test: Optional[bool] = False,
                 **replacements) -> str:
        """
        Generates a response from the LLM model.

        Parameters:
        - model_prompt_dir (str): The directory of the model prompt.
        - prompt_name (str): The name of the prompt.
        - prefix (str, optional): A prefix to match before the numbered list.
        - numbered_list (bool, optional): If True, the response will be
          extracted as a numbered list.
        - remove_number (bool, optional): If True, the numbered list will
          be cleaned of numbering.
        """
        system_prompt, user_prompt = self.prompt_model.process_prompt(
            model_name=model_prompt_dir,
            prompt_name=prompt_name,
            **replacements
        )
        response_content = ''
        try:
            stream_response = self.client.chat.stream(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.01,
                max_tokens=4096,
            )
            for chunk in stream_response:
                if chunk.data.choices:
                    response_content += chunk.data.choices[0].delta.content
                else:
                    return "No response generated."
        except Exception as e:
            print(f'Error: {e}')
        if test:
            print(response_content)
        # post processing
        if "```" in response_content and not numbered_list:
            # extract code from code blocks
            response_content = self.extract_code(response_content)
        elif numbered_list and (prefix is not None
                                or re.search(r'\d+\.', response_content)):
            response_content = self.extract_numbered_list(response_content,
                                                          prefix,
                                                          remove_number)
        return response_content


class Ollama(GenerativeModel):
    def __init__(self, model_name, prompt_model=None):
        super().__init__(model_name, prompt_model)
        self.model_name = model_name.replace('-', ':')
        if prompt_model is None:
            self.prompt_model = PromptModel()

    def generate(self,
                 model_prompt_dir: str,
                 prompt_name: str,
                 prefix: Optional[str] = None,
                 numbered_list: Optional[bool] = False,
                 remove_number: Optional[bool] = False,
                 test: Optional[bool] = False,
                 **replacements) -> str:
        """
        Generates a response from the LLM model.

        Parameters:
        - model_prompt_dir (str): The directory of the model prompt.
        - prompt_name (str): The name of the prompt.
        - prefix (str, optional): A prefix to match before the numbered list.
        - numbered_list (bool, optional): If True, the response will be
          extracted as a numbered list.
        - remove_number (bool, optional): If True, the numbered list will
          be cleaned of numbering.
        """
        system_prompt, user_prompt = self.prompt_model.process_prompt(
            model_name=model_prompt_dir,
            prompt_name=prompt_name,
            **replacements
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        options = {
             "temperature": 0.01,
             "num_predict": 4096
        }
        response_content = ""
        try:
            stream_response = ollama.chat(self.model_name,
                                          messages=messages,
                                          options=options,
                                          stream=True)
            for chunk in stream_response:
                if 'message' in chunk and 'content' in chunk['message']:
                    response_content += chunk['message']['content']
                else:
                    print("No response generated.")
                    break

        except Exception as e:
            print('Error:', e)
            return
        if test:
            print(response_content)
        # post processing
        if "```" in response_content and not numbered_list:
            # extract code from code blocks
            response_content = self.extract_code(response_content)
        elif numbered_list and (prefix is not None
                                or re.search(r'\d+\.', response_content)):
            response_content = self.extract_numbered_list(response_content,
                                                          prefix,
                                                          remove_number)
        return response_content
