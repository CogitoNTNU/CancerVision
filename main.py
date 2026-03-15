import sys

import torch

# if slurm output files exist, print their contents

stdout_file="slurm_outputs/output_combined.txt"
stderr_file="slurm_outputs/output_combined.err"

def main():
    # Probe CUDA with a tiny operation because availability alone can be misleading
    # on nodes where device architecture is unsupported by the installed torch build.
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU.")
        device = torch.device("cpu")

    # Example tensor operation
    a = torch.tensor([1.0, 2.0, 3.0], device=device)
    b = torch.tensor([4.0, 5.0, 6.0], device=device)
    c = a + b

    print("Result of tensor addition:", c)
    print("Device used for computation:", c.device)
    # write to stdout and stderr
    print("This is a message to stdout.")
    print("This is a message to stderr.", file=sys.stderr)


if __name__ == "__main__":
    main()  