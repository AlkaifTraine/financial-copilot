"""Company identity resolution: free-text name -> ticker, exchange, CIK."""

from .company import Company, resolve_company

__all__ = ["Company", "resolve_company"]
