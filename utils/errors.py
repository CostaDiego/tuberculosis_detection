class Error(Exception):
    """Base class for other exceptions"""

    pass


class DatasetDescriptionError(Error):
    """Raised when the dataset description is not valid"""

    pass

class DataframeGenerationError(Error):
    """Raised when the dataframe generation fails"""

    pass