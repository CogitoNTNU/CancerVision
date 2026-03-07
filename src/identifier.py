import os



class Cache:
    # Static variables
    __data = {}

    # Methods
    @classmethod
    def identifier(cls: 'Cache', key: tuple) -> (str | None): 
        return cls.__data.get(key)

    @classmethod
    def enter(cls: 'Cache', cacheKey: tuple, value: str) -> None: 
        cls.__data[cacheKey] = value

@staticmethod
def id(
    modelType: str,
    trainingBatchSize: int,
    validationBatchSize: int,
    epochs: int,
    versions: str = "logs/training",
    extension: str | None = ".log"
) -> str:
    # Check if the identifier has already been generated and cached, if so return it
    cacheKey = (modelType, trainingBatchSize, validationBatchSize, epochs)
    if (cachedID := Cache.identifier(cacheKey)): 
        print(f"Identifier retrieved from cache: {cachedID}")
        return cachedID

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

    # Cache the generated identifier for future use and return it
    identifier = f"{pattern}_{version}"
    print(f"Generated new identifier: {identifier}")
    Cache.enter(cacheKey, identifier)

    return identifier