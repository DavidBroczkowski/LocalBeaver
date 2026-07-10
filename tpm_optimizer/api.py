

def run_iteration(
    beaver_run_kwargs: dict(),
    num_instances: int,
    num_children: int,
    code_paths: List[str],
    dataset: str,
):
    """
    Runs one iteration of the optimization program

    Inputs:
        - beaver_run_kwargs: a dictionary containing the arguments needed to run BEAVER, excluding 'model' and 'prompts'
        - num_instances: an int with the number of total inputs/instances to use for optimization
        - num_children: an int with the maximum number of children, or modified programs, allowed at any time
        - code_paths: a List of str that contains the paths to the programs to optimize
        - dataset: a str, the path to the dataset to optimize on
    Outputs:
        - a List of str containing the paths to new optimized programs

    """

    # run code on LLaMa agent and get summary
    

    #reminder, need to define model and prompts for beaver kwargs before feeding into beaver