import logging
import os


class TrainingLogger:
    def __init__(self, 
        trainingBatchSize: int,
        validationBatchSize: int, 
        epochs: int,
        **hyperparameters
    ) -> None:
        """
        initialize the TrainingLogger with the specified training configuration parameters and hyperparameters.
        
        Args:
            trainingBatchSize: The batch size used for training.
            validationBatchSize: The batch size used for validation.
            epochs: The total number of training epochs.
            **hyperparameters: Arbitrary keyword arguments representing other training configuration hyperparameters.
        
        Returns:
            None

        ---

        Example:
            ```python
            logger.logTrainingConfig(
                trainingBatchSize=32,
                validationBatchSize=16,
                epochs=10,
                optimizer="Adam",
                loss_function="CrossEntropyLoss"
            )
            ```
        """
        # Hyperparameters
        self.hyperparameters = {
            "training_batch_size": trainingBatchSize,
            "validation_batch_size": validationBatchSize,
            "epochs": epochs
        } | hyperparameters

        # Log file
        LOG_DIRECTORY = "logs/training"
        LOG_EXTENSION = ".log"
        os.makedirs(LOG_DIRECTORY, exist_ok=True)

        baseName = f"B{trainingBatchSize}-V{validationBatchSize}-E{epochs}_"
        existingFiles = [                                                               
            f                                                                           # List all files in the  
            for f in os.listdir(LOG_DIRECTORY)                                          # LOG_DIRECTORY that start with
            if (f.startswith(baseName) and f.endswith(LOG_EXTENSION))                   # the baseName and end with .log
        ]

        version = len(existingFiles)
        filename = f"{baseName}{version}{LOG_EXTENSION}"
        filepath = os.path.join(LOG_DIRECTORY, filename)

        # Logger setup
        self.logger = logging.getLogger(f"training.{filename}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()

        fileHandler = logging.FileHandler(filepath, mode="w", encoding="utf-8")
        fileHandler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(fileHandler)

        # Log the training configuration parameters
        self.__logTrainingConfig()


    def __logTrainingConfig(self) -> None:
        self.logger.info("[Training configuration]")

        for key, value in self.hyperparameters.items():
            self.logger.info(f"{key.replace('_', ' ').title():<23}: {value}")

        # Log a separator line
        self.logger.info(f"\n\n{'-' * 20}")
        # self.logger.info("\n\n")


    def log(self, message: str):
        self.logger.info(message)


    def logEpochMetrics(self, /, epoch: int, **metrics) -> None:
        """
        Log the metrics for a specific training epoch.
        
        Args:
            epoch: The epoch number.
            **metrics: Arbitrary keyword arguments representing the metrics to log for the epoch.
            
        Returns:
            None
        """
        self.logger.info(f"[Epoch {epoch} metrics]")


        for key, value in metrics.items():
            self.logger.info(f" - {key.replace('_', ' ').title()}: {value}")


    def logValidationResults(self, /, currentEpoch: int, **results) -> None:
        """
        Log the validation results for a specific training epoch.
        
        Args:
            currentEpoch: The current epoch number.
            **results: Arbitrary keyword arguments representing the validation results to log for the epoch.
            
        Returns:
            None
        """
        self.logger.info(f"[Epoch {currentEpoch} validation results]")

        for key, value in results.items():
            self.logger.info(f" - {key.replace('_', ' ').title()}: {value}")


    def logCompletion(self, best_metric: float, best_metric_epoch: int) -> None:
        """
        Log the completion of the training process along with the best metric and the epoch it was achieved.
        
        Args:
            best_metric: The best metric value achieved during training.
            best_metric_epoch: The epoch number at which the best metric was achieved.
            
        Returns:
            None
        """
        self.logger.info(f"Training completed. Best metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
