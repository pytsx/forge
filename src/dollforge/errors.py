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
