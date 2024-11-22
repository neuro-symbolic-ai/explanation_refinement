# initialisation code from https://github.com/inpefess/isabelle-client
# and https://github.com/lanzhang128/retrieval_augmented_autoformalization
import time
import json
import re
import yaml

from isabelle_client import start_isabelle_server, get_isabelle_client

from models.autoformalisation_model import AutoFormalisationModel


class IsabelleSolver:
    def __init__(self,
                 llm,
                 isabelle_session='HOL',
                 port=7777,
                 isabelle_name='test',
                 watchdog_timeout=65,
                 dirs='../../Isabelle2023'):
        self.llm = llm
        self.autoformalisation_model = AutoFormalisationModel(llm)
        self.isabelle_name = isabelle_name
        self.port = port
        self.log_file = 'server.log'
        self.session_name = isabelle_session
        self.dirs = dirs
        self.verbose = True
        self.options = None
        self.watchdog_timeout = watchdog_timeout
        self._init_client()
        self._init_session()
        self.isabelle_dir = self._get_isabelle_dir()

    # get the path which saves isabelle theories
    def _get_isabelle_dir(self):
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        return config['isabelle']['master_dir']

    # init isabelle server
    def _init_client(self):
        server_info, _ = start_isabelle_server(
            name=self.isabelle_name, port=self.port, log_file=self.log_file
        )
        self.isabelle = get_isabelle_client(server_info)

    # init isabelle session (HOL, ZF, HOL-Proof, ...)
    def _init_session(self):
        self.isabelle.session_build(
            session=self.session_name, dirs=self.dirs,
            verbose=self.verbose, options=self.options
        )
        self.start_id = self.isabelle.session_start(session=self.session_name)

    # get isabelle response
    def _get_response(self, theories, master_dir):
        start_time = time.time()
        isabelle_response = self.isabelle.use_theories(
            session_id=self.start_id,
            theories=theories,
            master_dir=master_dir,
            watchdog_timeout=self.watchdog_timeout
        )
        solving_time = time.time() - start_time
        return isabelle_response, solving_time

    def get_isabelle_syntax_output(self,
                                   isabelle_code: str,
                                   theory_name: str,
                                   data_name: str,
                                   model_name: str,
                                   iteration: int) -> bool:
        has_syntax_error = True
        # check and refine any syntatic errors in the code
        for i in range(3):
            has_inner_syntax_error = False
            has_contradiction_error = False
            error_detail = []
            inner_code = ''
            contradiction_code = ''
            inference_time = 9999
            # check inner and contradiction error
            master_dir = \
                f'formalisation/{data_name}/{model_name}/{theory_name}'
            (has_inner_syntax_error,
             has_contradiction_error,
             error_detail,
             inner_code,
             contradiction_code,
             inference_time
             ) = self._check_syntax_error(theory_name,
                                          data_name,
                                          model_name,
                                          iteration,
                                          master_dir,
                                          isabelle_code)
            # if has inner syntax error, refine it
            if has_inner_syntax_error:
                print("Has inner syntax error at syntax "
                      f"refinement iteration {i}")
                refined_code = \
                    self.autoformalisation_model.fix_inner_syntax_error(
                      isabelle_code, error_detail, inner_code
                    )
                self.autoformalisation_model.save_formalised_kb(
                    refined_code,
                    theory_name,
                    data_name,
                    model_name,
                    iteration
                )
                isabelle_code = self.autoformalisation_model.code
                continue
            # if has contradition syntax error, refine it
            if has_contradiction_error:
                print("Has contradition syntax error at syntax "
                      f"refinement iteration {i}")
                refined_code = \
                    self.autoformalisation_model.fix_contradiction_error(
                      isabelle_code, contradiction_code
                    )
                self.autoformalisation_model.save_formalised_kb(
                    refined_code,
                    theory_name,
                    data_name,
                    model_name,
                    iteration
                )
                isabelle_code = self.autoformalisation_model.code
                continue
            if not has_inner_syntax_error and not has_contradiction_error:
                has_syntax_error = False
                break
            else:
                continue

        self.autoformalisation_model.save_formalised_kb(
            isabelle_code,
            theory_name,
            data_name,
            model_name,
            iteration
        )
        return has_syntax_error

    # using isabelle client to call isabelle to check
    def _check_syntax_error(self, theory_name, data_name, model_name,
                            iteration, master_dir, isabelle_code):
        isa_code_lines = isabelle_code.split('\n')
        for i, line in enumerate(isa_code_lines):
            if line.strip().startswith('shows'):
                start = line.index('"')
                end = line.rindex('"')
                isa_code_lines[i] = line[:start] + 'False' + line[end+1:]
                break
        check_syntax_error_code = '\n'.join(isa_code_lines)
        pattern = r'(proof -).*?(qed)(?!.*qed)'
        check_syntax_error_code = re.sub(pattern, r'  sledgehammer\n  oops',
                                         check_syntax_error_code,
                                         flags=re.DOTALL)
        directory = f'formalisation/{data_name}/{model_name}/{theory_name}'
        with open(f'{directory}/{theory_name}_{iteration}.thy', 'w') as f:
            f.write(check_syntax_error_code)

        theories_name = [f'{theory_name}_{iteration}']
        isabelle_response, solving_time = \
            self._get_response(theories_name, master_dir)
        # print(f"Isabelle response: {isabelle_response}")
        has_inner_syntax_error = False
        has_contradiction_error = False
        error_details = []
        lines = []
        inner_code = ''
        error_code_detail = []
        tactic_list = []
        tactic_messages = []
        contradiction_code = ''
        explanations = []
        found_explanations = {}
        finished_response = next((item for item in isabelle_response
                                  if item.response_type == 'FINISHED'), None)
        # Error Keywords
        error_keywords = ["Type unification failed", "Inner lexical error",
                          "Outer syntax error", "Inner syntax error",
                          "Outer lexical error", "Malformed command syntax",
                          "Undefined type name"]
        # Warning Keywords
        warning_keywords = ["Introduced fixed type variable"]
        if finished_response is not None:
            response_body = json.loads(finished_response.response_body)
            # Handling errors
            if response_body.get('errors'):
                for error in response_body['errors']:
                    message = error['message']
                    position = error['pos']
                    line = position['line']
                    if any(keyword in message for keyword in error_keywords):
                        error_details.append(
                            f"Error on line {line}: {message}"
                        )
                        lines.append(line)
                        has_inner_syntax_error = True
            else:
                has_inner_syntax_error = False

            for node in response_body.get('nodes', []):
                for message in node.get('messages', []):
                    tactic_messages.append(message['message'])

            if all("no proof found" not in item.lower()
                   for item in tactic_messages):
                tactic_list = [item for item in tactic_messages
                               if "try this:" in item.lower()]
                for item in tactic_list:
                    matches = re.findall(r'explanation_\d+', item)
                    explanations.extend(matches)
                explanations = sorted(set(explanations),
                                      key=lambda x: int(x.split('_')[1]))
                isabelle_code_lines = isabelle_code.split('\n')
                found_explanations = {}
                for line in isabelle_code_lines:
                    for exp in explanations:
                        if exp in line:
                            found_explanations[exp] = line.strip()
                if explanations != [] and found_explanations != {}:
                    contradiction_code = \
                        '\n\n'.join(found_explanations.values())
                has_contradiction_error = True
            else:
                has_contradiction_error = False

            # Handling warnings
            nodes = response_body.get('nodes', [])
            for node in nodes:
                messages = node.get('messages', [])
                for message in messages:
                    if message['kind'] == 'warning':
                        warning_message = message['message']
                        position = message['pos']
                        line = position['line']
                        if any(keyword in warning_message
                               for keyword in warning_keywords):
                            error_details.append(
                                f"Error on line {line}: {message}"
                            )
                            lines.append(line)
                            has_inner_syntax_error = True
        else:
            print("wrong theory name")
            return False, False, [9999], '', '', 9999
        inner_code = ''
        isabelle_lines = isabelle_code.splitlines()
        for line_number in lines:
            index = line_number - 1
            if index < len(isabelle_lines):
                line_text = isabelle_lines[index].strip()

                if "axiomatization where" in line_text:
                    if index + 1 < len(isabelle_lines):
                        inner_code = (inner_code +
                                      isabelle_lines[index + 1].strip() +
                                      '\n'
                                      if inner_code != ''
                                      else
                                      isabelle_lines[index + 1].strip()
                                      + '\n')
                elif "hypothesis" in line_text:
                    if index + 1 < len(isabelle_lines):
                        for i in range(1, 5):
                            if index + i < len(isabelle_lines):
                                inner_code += \
                                    isabelle_lines[index + i].strip() + '\n'
                else:
                    inner_code = inner_code + line_text+'\n'
        error_code_detail = "\n".join(f"{index}. {item}" for index, item in
                                      enumerate(error_details, start=1))

        return has_inner_syntax_error, has_contradiction_error, \
            error_code_detail, inner_code, contradiction_code, \
            solving_time

    def solve(self,
              theory_name,
              data_name,
              model_name,
              iteration,
              isabelle_code,
              explanation
              ):
        master_dir = \
                f'formalisation/{data_name}/{model_name}/{theory_name}'
        theories_name = [f'{theory_name}_{iteration}']
        proof_file_path = f'{master_dir}/{theory_name}_{iteration}.thy'
        isabelle_response, inference_time = \
            self._get_response(theories_name, master_dir)
        error_details = []
        error_lines = []
        stuck_error_line = []
        error_code = ''
        is_valid = False
        proof_line_number = None
        qed_line_number = None
        percentage_decimal = None
        finished_response = next((item for item in isabelle_response
                                  if item.response_type == 'FINISHED'), None)
        if finished_response is not None:
            response_body = json.loads(finished_response.response_body)

            data_dict = json.loads(finished_response.response_body)

            percentage = None
            if data_dict.get('nodes') and len(data_dict['nodes']) > 0:
                if 'percentage' in data_dict['nodes'][0]['status']:
                    percentage = int(
                        data_dict['nodes'][0]['status']['percentage']
                        )

            print(f'the finishing percentage is: {percentage}')

            # Handling errors
            if response_body.get('errors'):
                for error in response_body['errors']:
                    message = error['message']
                    position = error['pos']
                    line = position['line']
                    error_details.append(f"Error on line {line}: {message}")
                    error_lines.append(line)
                    is_valid = False
            elif percentage != 100:
                is_valid = False
                isabelle_file_path = proof_file_path
                with open(isabelle_file_path, 'r') as file:
                    for i, line in enumerate(file, start=1):
                        if "proof -" in line and proof_line_number is None:
                            proof_line_number = i
                        if "qed" in line and qed_line_number is None:
                            qed_line_number = i

                percentage_decimal = percentage / 100
                if percentage_decimal != 1 and error_details == []:
                    if (qed_line_number is not None and
                       proof_line_number is not None):
                        difference = qed_line_number - proof_line_number - 1
                        number = round(difference * percentage_decimal)
                    else:
                        number = 0
                if proof_line_number is not None:
                    stuck_error_line.append(proof_line_number + number)
            else:
                is_valid = True

            # Handling warnings
            nodes = response_body.get('nodes', [])
            for node in nodes:
                messages = node.get('messages', [])
                for message in messages:
                    if message['kind'] == 'warning':
                        warning_message = message['message']
                        position = message['pos']
                        line = position['line']
                        error_details.append(
                            f"Error on line {line}: {warning_message}"
                        )
                        error_lines.append(line)
                        is_valid = False
        else:
            print("Wrong theory name")
            return False, '',  9999

        explanation = explanation.split('.')
        numbered_explanation = [
            f'{index + 1}. {sentence.strip()}'
            for index, sentence in enumerate(explanation)
            if sentence.strip()
        ]
        explanation = "\n".join(numbered_explanation)
        error_code = ''
        isabelle_code_lines = isabelle_code.split('\n')
        if error_lines != []:
            for current_line_number, line in enumerate(isabelle_code_lines,
                                                       start=1):
                if current_line_number == error_lines[0]:
                    error_code = line.strip()

        if 'axiomatization' in error_code:
            for current_line_number, line in enumerate(isabelle_code_lines,
                                                       start=1):
                if current_line_number == error_lines[0]+1:
                    error_code = line.strip()
        if stuck_error_line != []:
            number = stuck_error_line[0]
            specific_line_content_first = \
                isabelle_code_lines[number - 1].strip()
            specific_line_content_second = \
                isabelle_code_lines[number - 2].strip()
            error_code = (error_code+specific_line_content_first+'\n' +
                          specific_line_content_second +
                          '\nstuck at these two lines')

        if error_code == '' and is_valid is False:
            print('syntatic error in the isabelle code, go to next iteration')
            explanation = re.sub(r'\d+\.\s*', '', explanation)
            return False, 'no', 9999

        return is_valid, error_code, inference_time

    def shutdown(self):
        self.isabelle.session_stop(session_id=self.start_id)
        self.isabelle.shutdown()
        print('Isabelle session is shut down.')
