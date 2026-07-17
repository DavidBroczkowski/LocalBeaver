import beaver
import json
import math
from pathlib import Path
import random
import sys
import time
import whatthepatch

from tpm_optimizer.utils.pydantic_utils import Agent
from tpm_optimizer.utils.logging import Logger, MsgType 

logger = None

def _write_to_file(path: str, msg: str):
    """
    Writes a given message to a file - CAUTION: Overwrites existing data

    Inputs:
        - path: a str containing the path for the file
        - msg: a str containing the message to write
    """
    with open(path, 'w', encoding='utf-8') as f:
        f.write(msg)

def initialize_agents(
        model: str, 
        num_children: int,
        max_tokens: int,
        base_url: str,
        api_key: str | None = None, 
        timeout: float = 600.0
    ):
    """
    Initializes the agents used for the optimization algorithm via the Agent class

    Inputs:
        - model: a str containing the name of the model to use
        - num_children: an int with the maximum number of children, or modified programs, allowed at any time
        - max_tokens: the maximum number of completion tokens allowed for the model
        - base_url: a str, the URL to call for the API
        - api_key: a str containing the API Key
        - timeout: a float with the number of seconds before timing out the request
    """

    code_sum_prompt = "You are an expert in deriving information flow in code.\n" \
    "You will receive a Python program that is a transformer network written out as Python code.\n" \
    "You will first try to understand how information flows through the transformer.\n" \
    "Do not merely summarize what the code does, but why it is written as it is.\n" \
    "Determine what attention layer or head does based on the code given and what information is gained at each layer. \n" \
    "Summarize your findings in a concise format. Do not output anything other than the summary. \n" \
    "The next message will be the code to analyze.\n"

    results_prompt = "You are an expert at finding trends in neural network verification output.\n" \
    "Below are some of the output from a branch and bound approach to neural network verification.\n" \
    "First, attempt to understand what task is being performed. Then, analyze the trends you see." \
    "DO NOT MAKE ANY COMMENTS ABOUT THE VERIFICATION METHOD USED." \
    "Instead, look at the results and how much probability is assigned to incorrect or correct output."\
    "Think about why this could have happened, such as lots of repeated integers, too many unique integers, long input length, etc." \
    "Once done, create a summary of these trends. The next message will be the data to analyze.\n"

    alterations_prompt = "You are an expert in writing Python code and in transformers related to machine learning.\n" \
    "You will be given Python code that represents a transformer network.\n" \
    "You will be given a summary of how information flows through this transformer. Use this to understand the code and structure.\n" \
    "Additionally, this transformer has been fed through a branch and bound verification algorithm.\n" \
    "A summary has been produced of where the network excelled and where it failed.\n" \
    "Your task is to use these summaries to produce alterations to the code, and therefore, alterations to the transformer itself.\n" \
    "You may ONLY change one logical statement branch per alteration. For example, \n"\
    "if position in {0, 3, 4, 7}:\n" \
    "return token == \"3\"\n"\
    "may be changed to\n" \
    "if position in {0, 3, 5, 7}:\n" \
    "return token == \"3\"\n"\
    "Do not make more than one logical branch change per alteration and do not alter any other code in the file. \n" \
    f"You may create only up to {num_children} alterations.\n" \
    "Format your response as plain text with one unified diff after the other.\n" \
    "An example of a unified diff is:\n"
    "--- a/file.py\n" \
    "+++ b/file.py\n" \
    "@@ -10,7 +10,7 @@\n" \
    "def my_function():\n" \
    "-    x = 1\n" \
    "+    x = 2\n" \
    "    return x\n" \
    "In your unified diffs, you MUST include the path of the file at the top. YOU MUST INCLUDE THE INDICES OF ThE LINES YOU ARE ALTERING AS SHOWN IN THE EXAMPLE." \
    "Respond only with these unified diffs. Include 3 lines of context around each change.\n" \
    "Do not include any explanation outside the diff block.\n" \
    "The next message will contain the code summary, verification results summary, and the code itself.\n"

    Code_Sum_Agent = Agent(
        model=model,
        system=code_sum_prompt,
        max_tokens=max_tokens,
        api_key=api_key,
        temperature=1,
        timeout=timeout,
        base_url=base_url,
    )

    Results_Agent = Agent(
        model=model,
        system=results_prompt,
        max_tokens=max_tokens,
        api_key=api_key,
        temperature=1,
        timeout=timeout,
        base_url=base_url,
    )

    Alterations_Agent = Agent(
        model=model,
        system=alterations_prompt,
        max_tokens=max_tokens,
        api_key=api_key,
        temperature=1,
        timeout=timeout,
        base_url=base_url,
    )

    return Code_Sum_Agent, Results_Agent, Alterations_Agent

def _print_token_usage(
    CodeSumAgent: Agent,
    ResultsAgent: Agent,
    AlterationsAgent: Agent,
): 
    """
    Prints and logs how many tokens each Agent used

    Inputs: 
        - CodeSumAgent: an Agent Object that allows for API calls to an LLM to summarize code
        - ResultsAgent: an Agent Object that allows for API calls to an LLM to summarize verification results
        - AlterationsAgent: an Agent Object that allows for API calls to an LLM to create alterations on Transformer Program code
    """
    agent_names = ["Code Summary Agent", "Verification Results Agent", "Alteration Creation Agent"]
    agents = [CodeSumAgent, ResultsAgent, AlterationsAgent]

    for agent, agent_name in zip(agents, agent_names):
        token_msg = f"Token Usage for {agent_name}: Overall - {agent.total_tokens}, Prompt - {agent.prompt_tokens}, Completion - {agent.completion_tokens}, Reasoning - {agent.reasoning_tokens}"
        logger.log(token_msg)

def _prune_results(results: list[dict], k=25):
    """
    Prunes results to the top-k best and worst performing instances based on upper bounds

    Inputs:
        - results: a list of dictionaries containing BEAVER verification information of each instances
        - k: an int for how many results to prune to
    Outputs:
        - best_results, a list[dict] containing the top-k best performing instance dictionaries, and worst_results, a list[dict] containing the top-k worst. 
    """
    if k > len(results):
        logger.log(f"num_instances is greater than the length of the dataset. It will be set to the length of the dataset, {len(results)}", type=MsgType.WARN)
        inst_num = math.floor(len(results)/2)
    
    # set inst_num to half of k as we want k/2 of either side, best and worst
    inst_num = math.floor(k/2)

    sorted_results = sorted(results, key=lambda x: x["upper_bound"], reverse=True)
    return [], sorted_results[-1:]

def _get_result_logs(results: list[dict], log_folder: str):
    """
    Retrieves the entire log files associated with results.

    Inputs:
        - results: a list[dict] containing the last iteration details of verification, including the index of the instance
        - log_folder: a str containing the path to the folder containing the log files
    
    Output:
        - a list[str] containing the log files for each instance's verification output
    """
    indices = [result["idx"] for result in results]

    log_files = []
    for index in indices:
        file_path = log_folder + f"/{index}.jsonl"
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
        log_files.append(str(data))
    
    return log_files

def _build_iteration_dataset(best_results: list[dict], worst_results: list[dict], dataset: list[dict]):
    """
    Uses the pruned_results to build the dataset GLM will summarize

    Inputs:
        - best_results: a list[dict] containing the top-k best performing instance dictionaries
        - worst_results: a list[dict] containing the top-k worst performing instance dictionaries
        - dataset: a list[dict] where each dictionary contains information about the prompt, its index, its labels
    Output:
        - a list[dict] containing the best and worst prompt dictionaries from dataset
    """

    indices = set(result["idx"] for result in (best_results + worst_results))
    return [prompt for prompt in dataset if prompt["idx"] in indices]

def _apply_alterations(
    code_path: str, 
    alterations: str
):
    """
    Uses whatthepatch to reconstruct transformer program code based on the alterations, unified diffs, given by the LLM

    Inputs:
        - code_path: a str containing the path of the transformer program code
        - alterations: a str containing unified diffs for the code at code_path
    Output:
        - a list[str] containing the reconstructed code, one for each alteration
    """

    with open(code_path, 'r', encoding = 'utf-8') as f:
        code = f.read()
    
    print(f"[DEBUG] ALTERATIONS: {alterations}")
    try:
        diffs = [x for x in whatthepatch.parse_patch(alterations)]
        print(diffs)
        reconstructions = [whatthepatch.apply_diff(diff, code) for diff in diffs]
    except Exception as e:
        import traceback
        logger.log("An error occured when attempting to patch together the alterations.", type=MsgType.ERROR)
        logger.log(e, type=MsgType.ERROR)
        traceback.print_exc()
    
    return reconstructions

def _rank_code(
    code_info: list[dict],
):
    """
    Sorts the code paths from best to worst based on the average upper bound of the verification results.

    Inputs:
        - code_info: a list[dict] where the dictionary contains the code's path, verification output, and log folder path
    Output:
        - a list[dict] containing the code instances sorted by highest average upper bound to lowest
    """

    return sorted(code_info, key=lambda x: x["summary"]["avg_ub"], reverse=True)



def _summarize_code(
    CodeSumAgent: Agent,
    code_path: str,
    dry_run: bool,
):
    """
    Summarizes how information flows through a transformer program via Qwen 2.5

    Inputs:
        - CodeSumAgent: an Agent Object that allows for API calls to an LLM to summarize code
        - code_path: a str containing the path to the transformer program code
        - dry_run: a bool, when True, does not call any API and feeds artificial data through the algorithm
    Output:
        - a str containing the LLMs summary response
    """
    with open(code_path, 'r', encoding = 'utf-8') as f:
        code = f.read()

    if dry_run:
        logger.log(f"System Prompt: {CodeSumAgent.system}", console=False, type=MsgType.DEBUG)
        logger.log(f"User Prompt: {json.dumps(code)}", console=False, type=MsgType.DEBUG)
        logger.log(f"Request Payload: {CodeSumAgent.make_request_payload(json.dumps(code))}", console=False, type=MsgType.DEBUG)
        logger.log("Dry run, skipping summarization")
        return "DRY RUN"

    lines = code.splitlines()
    payload = CodeSumAgent.make_request_payload(json.dumps(code))
    try:
        response_data = CodeSumAgent.post_chat_completion(payload)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return None
    response = CodeSumAgent.extract_completion(response_data)

    logger.log(f"Results Agent gave response: {response}")
    logger.log(f"Full Response: {response_data}", console=False)

    return response
    

def _summarize_results(
    ResultsAgent: Agent,
    dataset: list[str],
    dry_run: bool,
):
    """
    Summarizes the results of a verification run via Qwen 2.5

    Inputs:
        - ResultsAgent: an Agent Object that allows for API calls to an LLM to summarize verification results
        - dataset: a list[str] containing the full verification output of the instances to summarize
        - dry_run: a bool, when True, does not call any API and feeds artificial data through the algorithm
    Output:
        - a str containing the LLMs summary response
    """

    if dry_run: 
        logger.log(f"System Prompt: {ResultsAgent.system}", console=False, type=MsgType.DEBUG)
        logger.log(f"User Prompt: {json.dumps(dataset)}", console=False, type=MsgType.DEBUG)
        logger.log(f"Request Payload: {ResultsAgent.make_request_payload(json.dumps(dataset))}", console=False, type=MsgType.DEBUG)
        logger.log("Dry run, skipping summarization")
        return "DRY RUN"
    
    payload = ResultsAgent.make_request_payload(json.dumps(dataset))
    try:
        response_data = ResultsAgent.post_chat_completion(payload)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    response = ResultsAgent.extract_completion(response_data)

    logger.log(f"Results Agent gave response: {response}")
    logger.log(f"Full Response: {response_data}", console=False)

    return response

def _create_alterations(
    AlterationsAgent: Agent,
    code_path: str,
    code_summary: str, 
    results_summary: str, 
    dry_run: bool,
):
    """
    Uses the summaries to create alterations to the code at code_path. Restricts the number of alterations to num_children

    Inputs:
        - AlterationsAgent: an Agent Object that allows for API calls to an LLM to create alterations on Transformer Program code
        - code_path: a str containing the path to the transformer program code
        - code_summary: a str containing the LLM's summary response to the code
        - results_summary: a str containing the LLM's summary response to the verification results
        - num_children: an int with the maximum number of alterations to produce
        - dry_run: a bool, when True, does not call any API and feeds artificial data through the algorithm
    Output:
        - a str containing the output of the model - should be unified diffs
    """
    
    with open(code_path, 'r', encoding = 'utf-8') as f:
        code = f.read()

    # add indices to the lines of code, needed for the unified diffs
    code_array = code.splitlines()
    code_llm = []
    for i, (code_line) in enumerate(code_array):
        code_llm.append([f"{i+1} {code_line}"])

    # use splitlines there because it might help the LLM to find the line indices
    user_prompt = f"Code Path: {code_path} \n" \
    f"Code: {code_llm} \n" \
    f"Code Summary: {code_summary} \n" \
    f"Verification Results Summary: {results_summary} \n"

    if dry_run: 
        logger.log(f"System Prompt: {AlterationsAgent.system}", console=False, type=MsgType.DEBUG)
        logger.log(f"User Prompt: {user_prompt}", console=False, type=MsgType.DEBUG)
        logger.log(f"Request Payload: {AlterationsAgent.make_request_payload(user_prompt)}", console=False, type=MsgType.DEBUG)
        logger.log("Dry run, skipping alteration creation")

        lines = code.splitlines()
        # return a dummy change
        return (
            f"--- a/{code_path}\n"
            f"+++ b/{code_path}\n"
            "@@ -141,7 +141,7 @@\n"
            "         elif position in {4}:\n"
            "             return token == \"<s>\"\n"
            "         elif position in {5, 6}:\n"
            f"-{lines[143]}\n" # dynamically load the line to fit with whatthepatch
            f"+            return token == \"{random.randint(1,6)}\"\n"
            "\n"
            #FIXME: this line beneath was causing an error when attempting to apply the alteration with whatthepatch due to a mismatch to an empty string, why?
            #"     num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)\n"
        )
    payload = AlterationsAgent.make_request_payload(user_prompt)
    try:
        response_data = AlterationsAgent.post_chat_completion(payload)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    response = AlterationsAgent.extract_completion(response_data)

    logger.log(f"Alterations Agent gave response: {response}")
    logger.log(f"Full Response: {response_data}", console=False)

    return response

def _run_iteration(
    beaver_run_kwargs: dict,
    num_instances_llm: int,
    num_instances_verify: int,
    num_children: int,
    code_info: list[dict],
    dataset: list[dict],
    out_dir: Path,
    dry_run: bool,
    CodeSumAgent: Agent,
    ResultsAgent: Agent,
    AlterationsAgent: Agent,
):
    """
    Runs one iteration of the optimization program

    Inputs:
        - beaver_run_kwargs: a dictionary containing the arguments needed to run BEAVER, excluding 'model' and 'prompts'
        - num_instances_llm: an int with the number of total inputs/instances to use for input to the llm for summarization
        - num_instances_verify: an int with the number of total inputs/instances to use for verification of each alteration
        - num_children: an int with the maximum number of children, or modified programs, allowed at any time
        - code_info: a list[dict] where the dictionary contains the code's path, verification output, and log folder path
        - dataset: a list[dict] where each dictionary contains information about the prompt, its index, its labels
        - out_dir: a Path object where BEAVER run logs will be sent to
        - dry_run: a bool, when True, does not call any API and feeds artificial data through the algorithm
        - CodeSumAgent: an Agent Object that allows for API calls to an LLM to summarize code
        - ResultsAgent: an Agent Object that allows for API calls to an LLM to summarize verification results
        - AlterationsAgent: an Agent Object that allows for API calls to an LLM to create alterations on Transformer Program code
    Outputs:
        - a list[dict] containing the paths, verification output, and log dirs for the Python files produced during this iteration

    """
    top_paths = []

    for i, (info) in enumerate(code_info):
        code_path = info["path"]
        code_result = info["results"]
        code_log_dir = info["log_dir"]

        logger.log(f"Processing code_path: {code_path}, {i+1} out of {len(code_info)}...")

        # build log files to summarize
        best_results, worst_results = _prune_results(code_result, num_instances_llm)
        results_logs = _get_result_logs(best_results + worst_results, code_log_dir)

        # run code on GLM agent and get summary
        code_summary = _summarize_code(CodeSumAgent, code_path, dry_run)

        # run code on GLM agent for each code_path in code_paths to get a List of summaries
        results_summary = _summarize_results(ResultsAgent, results_logs, dry_run)

        # run code on GLM agent for each code_path in code_paths to optimize them
        alterations = _create_alterations(AlterationsAgent, code_path, code_summary, results_summary, dry_run)

        # construct our new Python files
        # normalizes the text as cannot parse the ` character and is commonly returned by the model
        diff_text = alterations.strip()
        if diff_text.startswith("```"):
            diff_text = "\n".join(diff_text.split("\n")[1:])  # remove first line
        if diff_text.endswith("```"):
            diff_text = "\n".join(diff_text.split("\n")[:-1])  # remove last line

        reconstructions = _apply_alterations(code_path, diff_text)

        # write these files to out_dir
        out_dir.mkdir(exist_ok=True)
        file_paths = []

        for j, (reconstruction) in enumerate(reconstructions):
            file_path = out_dir / f"{i}_{j}.py"
            file_paths.append(file_path)
            _write_to_file(file_path, "\n".join(reconstruction))
        
        # run verification
        logger.log(f"Received {len(file_paths)} alterations, verifying alterations...")

        iteration_results = []
        verification_prompts = random.choices(dataset, k=num_instances_verify)
        for j, (file_path) in enumerate(file_paths):
            beaver_run_kwargs["log_dir"] = out_dir / f"{i}_{j}"
            beaver_response = beaver.run(
                prompts=verification_prompts,
                model=str(code_path),
                **beaver_run_kwargs,
            )
            print(beaver_response["results"])

            # add the file path to the dictionary and then store it
            beaver_response["path"] = file_path
            iteration_results.append(beaver_response)
        
        # rank the files
        ranked_code_info = _rank_code(iteration_results)

        # store top k for now, no need to store more than k because we would prune them later anyways
        top_paths += ranked_code_info[:num_children]

        logger.log(f"Processed {code_path} and added paths")
    
    # sort the entire list and reduce to k
    top_paths = _rank_code(top_paths)[:num_children]

    return top_paths

def run(
    beaver_run_kwargs: dict,
    num_instances_llm: int,
    num_instances_verify: int,
    num_children: int,
    code_path: str,
    dataset: dict,
    max_steps: int,
    log_dir: str,
    dry_run: bool,
    llm_model: str,
    base_url: str,
    api_key: str | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
):
    """
    Runs an entire pass through the optimization algorithm - uses an LLM to optimize Transformer Program code based on the output of BEAVER verification

    Inputs:
        - beaver_run_kwargs: a dict containing the arguments needed to run the BEAVER verification algorithm
        - num_instances_llm: an int with the number of total inputs/instances to use for input to the llm for summarization
        - num_instances_verify: an int with the number of total inputs/instances to use for verification of each alteration
        - num_children: an int with the maximum number of children, or modified programs, allowed at any time
        - code_path: a str containing the path to the starting Transformer Program Python code
        - dataset: a list[dict] where each dictionary contains information about the prompt, its index, its labels
        - max_steps: an int with the number of iterations of alterations to perform in the algorithm
        - log_dir: a str containing the directory to send all output and logs to
        - dry_run: a bool, when True, does not call any API and feeds artificial data through the algorithm
        - llm_model: a str containing the name of the model to use
        - base_url: a str, the URL to call for the API
        - api_key: a str containing the API key to use for the models
        - max_tokens: an int with the maximum number of completion tokens allowed for each model
        - timeout: a float containing the number of seconds until an automatic timeout is called on the models
    
    Outputs:
        - a list[dict] containing the paths, verification output, and log dirs for the Python files produced during this optimization run
    """
    out_dir = Path(log_dir) / Path(code_path).stem / str(time.time())
    out_dir.mkdir(parents=True, exist_ok=False)

    global logger 
    logger = Logger(out_dir / "log.txt")


    # initialize models if not a dry run
    if not dry_run:
        CodeSumAgent, ResultsAgent, AlterationsAgent = initialize_agents(
            api_key=api_key, 
            model=llm_model, 
            num_children=num_children, 
            max_tokens=max_tokens, 
            base_url=base_url, 
            timeout=timeout,
        )
    else:
        logger.log("Running dry run, initializing models with no API Key...")
        CodeSumAgent, ResultsAgent, AlterationsAgent = initialize_agents(
            api_key=api_key, 
            model=llm_model, 
            num_children=num_children, 
            max_tokens=max_tokens, 
            base_url=base_url, 
            timeout=timeout,
        )

    logger.log("Running initial BEAVER verification...")

    # run initial beaver verification
    verification_prompts = random.choices(dataset, k=num_instances_verify)

    beaver_response = beaver.run(
        prompts=verification_prompts,
        model=code_path,
        **beaver_run_kwargs,
    )

    logger.log(f"Code Path: {code_path} completed with {len(beaver_response['results'])} results")

    iteration_num = 0

    # code info will hold each iterations code path, verification results, and verification summary
    code_info = [[{
        "path": code_path,
        "results": beaver_response["results"],
        "summary": beaver_response["summary"],
        "log_dir": beaver_response["log_dir"],
    }]]

    logger.log("Beginning iterations...")
    while iteration_num < max_steps:
        logger.log(f"Starting iteration {iteration_num+1} of {max_steps}...")

        code_info.append(_run_iteration(
            beaver_run_kwargs, 
            num_instances_llm, 
            num_instances_verify, 
            num_children, 
            code_info[iteration_num], 
            dataset,
            out_dir=out_dir / f"iter_{iteration_num+1}",
            dry_run=dry_run,
            CodeSumAgent=CodeSumAgent,
            ResultsAgent=ResultsAgent,
            AlterationsAgent=AlterationsAgent,
        ))

        logger.log("Finished iteration")

        iteration_num += 1

    # return best performing code
    logger.log("Flattening and sorting final results...")
    flattened_code = [item for iteration in code_info for item in iteration]
    sorted_code_info = _rank_code(flattened_code)[:num_children]

    _print_token_usage(CodeSumAgent, ResultsAgent, AlterationsAgent)

    return sorted_code_info