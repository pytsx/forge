class DomainError(Exception):
    """A safe, actionable error at a domain boundary."""


class NotFound(DomainError):
    pass


class Conflict(DomainError):
    pass


class InvalidInput(DomainError):
    pass


class DependencyUnavailable(DomainError):
    pass


class QualityLimitExceeded(DomainError):
    """A stage exhausted bounded self-correction without reaching its acceptance limit."""

