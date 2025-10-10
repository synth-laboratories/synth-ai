import logging


class ExcludeLangfuseMessagesFilter(logging.Filter):
    def filter(self, record):
        # Return False to exclude the record, True to include it
        message = record.getMessage()
        excluded_messages = [
            "No observation found in the current context",
            "No trace found in the current context",
            "Adding event to partition",
        ]
        return not any(msg in message for msg in excluded_messages)


# Configure root logger
root_logger = logging.getLogger()
root_logger.addFilter(ExcludeLangfuseMessagesFilter())
root_logger.setLevel(logging.ERROR)

# Configure langfuse logger
langfuse_logger = logging.getLogger("langfuse")
langfuse_logger.addFilter(ExcludeLangfuseMessagesFilter())
langfuse_logger.setLevel(logging.CRITICAL)
langfuse_logger.propagate = False

# Also configure the synth_sdk logger
synth_logger = logging.getLogger("synth_sdk")
synth_logger.addFilter(ExcludeLangfuseMessagesFilter())
synth_logger.setLevel(logging.ERROR)
synth_logger.propagate = False
