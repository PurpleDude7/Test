"""Exception hierarchy. Everything evalkit raises on purpose derives from EvalkitError."""


class EvalkitError(Exception):
    """Base class for all evalkit errors."""


class SuiteError(EvalkitError):
    """The suite file is missing, malformed, or fails validation."""


class RegistryError(EvalkitError):
    """An unknown provider or scorer name was requested."""


class ProviderError(EvalkitError):
    """A provider failed to produce a completion (network, auth, bad response...)."""


class NotSupportedError(ProviderError):
    """The provider does not implement the requested capability (e.g. embeddings)."""
