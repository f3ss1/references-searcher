class NotFittedError(Exception):
    """Exception raised when a machine learning model is used before being fit."""

    def __init__(self, message="This model is not fit yet. Please fit the model before using it."):
        self.message = message
        super().__init__(self.message)
