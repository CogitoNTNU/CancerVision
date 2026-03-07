from logger.train import TrainingLogger



logger = TrainingLogger(
    trainingBatchSize=32,
    validationBatchSize=16,
    epochs=10
)



logger.logEpochMetrics(
    epoch=1,
    training_loss=0.5,
    training_accuracy=0.8
)

logger.logValidationResults(
    currentEpoch=1,
    validation_loss=0.4,
    validation_accuracy=0.85
)


