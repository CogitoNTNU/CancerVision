import torch

def main():
    # Check if CUDA is available
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
if __name__ == "__main__":
    main()  