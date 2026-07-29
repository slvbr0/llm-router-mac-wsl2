"""Stub the `litellm` package so priority_router imports without the heavy dep.
litellm is only needed at runtime inside the container; unit tests exercise pure
routing logic. Injects fake modules into sys.modules before priority_router loads.
"""
import sys
import types

_litellm = types.ModuleType("litellm")
_integrations = types.ModuleType("litellm.integrations")
_custom_logger = types.ModuleType("litellm.integrations.custom_logger")
_logging = types.ModuleType("litellm._logging")


class CustomLogger:  # minimal stand-in for the base class
    pass


class _VerboseLogger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


_custom_logger.CustomLogger = CustomLogger
_logging.verbose_logger = _VerboseLogger()
_integrations.custom_logger = _custom_logger
_litellm.integrations = _integrations

sys.modules.setdefault("litellm", _litellm)
sys.modules.setdefault("litellm.integrations", _integrations)
sys.modules.setdefault("litellm.integrations.custom_logger", _custom_logger)
sys.modules.setdefault("litellm._logging", _logging)
