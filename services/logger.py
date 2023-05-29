import logging

# Create an instance of the logger
logger = logging.getLogger('ML360_COMPANY_STREAMING_DATA')

# Disable propagation from the root logger
logger.propagate = False

# Set the log level
logger.setLevel(logging.DEBUG)

# Configure the logging handler(s)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('app.log')

# Set the log level for each handler
console_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.WARNING)

# Create a formatter for the log messages
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')


# Add the formatter to the handlers
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

