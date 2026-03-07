import os


@staticmethod
def id(
    modelType: str,
    trainingBatchSize: int,
    validationBatchSize: int,
    epochs: int,
    versions: str = "logs/training",
    extension: str | None = ".log"
) -> str:
    # Generate a unique identifier based on the training configuration parameters
    pattern = f"{modelType}-B{trainingBatchSize}-V{validationBatchSize}-E{epochs}"
    previousVersions = [                                                               
        f                                                                               # List all files in the  
        for f in os.listdir(versions)                                                   # directory that start with the
        if (f.startswith(pattern) and (f.endswith(extension)                            # pattern and end with the extension
        if extension else True))                                                        # if a extension is provided
    ]

    # Add a version number to the identifier incase we retrain with the same parameters
    version = len(previousVersions)

    return f"{pattern}_{version}"