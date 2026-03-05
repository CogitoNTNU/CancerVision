import logging


class TrainingLogger:
    def __init__(self, logFile: str):
        logging.basicConfig(filename=logFile, level=logging.INFO, format="[%(asctime)s] %(message)s")
        self.logger = logging.getLogger()

    def log(self, message: str):
        self.logger.info(message)

    def logTrainingConfig(self, /, **hyperparameters) -> None:
        """
        Log the training configuration parameters.
        
        Args:
            **hyperparameters: Arbitrary keyword arguments representing training configuration hyperparameters.
        
        Returns:
            None

        ---

        Example:
            ```python
            logger.logTrainingConfig(
                training_batch_size=32,
                validation_batch_size=16,
                epochs=10,
                optimizer="Adam",
                loss_function="CrossEntropyLoss"
            )
            ```
        """
        # Log the training configuration parameters
        self.logger.info("Training configuration:")

        for key, value in hyperparameters.items():
            self.logger.info(f"{key.replace('_', ' ').title()}: {value:<24}")

    def logEpochMetrics(self, epoch: int, /, **metrics) -> None:
        """
        Log the metrics for a specific training epoch.
        
        Args:
            epoch: The epoch number.
            **metrics: Arbitrary keyword arguments representing the metrics to log for the epoch.
            
        Returns:
            None
        """
        # Log the epoch number
        self.logger.info(f"Epoch {epoch}:")

        for key, value in metrics.items():
            self.logger.info(f"  {key.replace('_', ' ').title()}: {value}")

    def logValidationResults(self, currentEpoch: int, /, **results) -> None:
        """
        Log the validation results for a specific training epoch.
        
        Args:
            currentEpoch: The current epoch number.
            **results: Arbitrary keyword arguments representing the validation results to log for the epoch.
            
        Returns:
            None
        """
        # Log the epoch number
        self.logger.info(f"Current epoch: {currentEpoch}")

        for key, value in results.items():
            self.logger.info(f"  {key.replace('_', ' ').title()}: {value}")

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
