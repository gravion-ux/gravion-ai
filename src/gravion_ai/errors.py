"""Exception types for gravion-ai.

Kept dependency-free so every layer (core, licensing, CLI) can raise and
catch the same errors without import cycles.
"""


class GravionError(Exception):
    """Base class for all gravion-ai errors."""


class ConfigError(GravionError):
    """Configuration is missing, unreadable, or invalid."""


class ProviderError(GravionError):
    """A model provider call failed."""


class ModuleError(GravionError):
    """A module could not be loaded or run."""


class CatalogError(GravionError):
    """The model catalog is missing, unreadable, or fails validation."""


class LicensingError(GravionError):
    """Base class for licensing / entitlement failures."""


class IdentityError(LicensingError):
    """Deployment identity is missing or corrupt."""


class LicenseError(LicensingError):
    """A license document is malformed."""


class BadSignatureError(LicenseError):
    """A signature did not verify."""


class LicenseExpiredError(LicenseError):
    """A license (or entitlement token) is past its expiry."""


class NotEntitledError(LicenseError):
    """The license/token does not cover the requested module."""


class ModuleTamperError(LicensingError):
    """A module's files do not match its signed manifest."""


class AuditChainError(LicensingError):
    """The audit log's hash chain does not verify."""


class SuspendedError(LicensingError):
    """Production use is suspended: no valid entitlement.

    Suspension is safe by design: it blocks inference and writes, but
    never deletes or locks away the deployment's data — export always
    remains available.
    """
