"""Real IBM runtime gateway factory — wired, but never executed without an account.

The factory builds a real :class:`acrpq.dashboard.ibm_runner.RuntimeGateway` (which
lazily imports ``qiskit-ibm-runtime``) ONLY when:

* the ``ACRPQ_IBM_RUNTIME_FACTORY_ENABLED`` flag is explicitly true, AND
* it is actually called (which the API only does AFTER every real-submission guard
  has passed — impossible with a fake artefact).

The IBM ``QiskitRuntimeService`` is constructed by an injectable
:class:`IbmServiceFactory` at the last moment; the account/token is read only by
IBM's own standard mechanisms — never passed as an argument, URL, run field, log or
export. Tests inject a **fake** service factory that reproduces the 0.47.0 surface;
no test ever constructs a real service, and the default flag keeps the seam closed
(503) so a bare server can never submit.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Protocol, runtime_checkable

_TRUE = frozenset({"1", "true", "yes", "on"})


class GatewayNotEnabled(RuntimeError):
    """The real runtime factory is not enabled/configured (fail-closed seam)."""


def runtime_factory_enabled(*, env: Callable[[str], str | None] = os.environ.get) -> bool:
    raw = env("ACRPQ_IBM_RUNTIME_FACTORY_ENABLED")
    return isinstance(raw, str) and raw.strip().lower() in _TRUE


@runtime_checkable
class IbmServiceFactory(Protocol):
    """Builds the IBM runtime service at the last moment (token via IBM's own means)."""

    def make_service(self) -> Any: ...


class _DefaultServiceFactory:
    """Constructs a real ``QiskitRuntimeService`` lazily. Never imported/built in tests."""

    def make_service(self) -> Any:  # pragma: no cover - requires a real IBM account
        from qiskit_ibm_runtime import QiskitRuntimeService  # lazy; reads the account itself

        return QiskitRuntimeService()


class RealIbmGatewayFactory:
    """A callable ``(backend_name) -> RuntimeGateway`` guarded by the runtime flag.

    Constructs nothing until called AND enabled. With an injected
    ``service_factory`` (a fake in tests) it never touches a real account; with the
    default factory it builds the real service only when the operator enabled it.
    """

    def __init__(self, *, env: Callable[[str], str | None] = os.environ.get,
                 service_factory: IbmServiceFactory | None = None) -> None:
        self._env = env
        self._service_factory = service_factory

    def enabled(self) -> bool:
        return runtime_factory_enabled(env=self._env)

    def __call__(self, backend_name: str) -> Any:
        if not runtime_factory_enabled(env=self._env):
            raise GatewayNotEnabled(
                "real IBM runtime factory is disabled "
                "(set ACRPQ_IBM_RUNTIME_FACTORY_ENABLED=true and configure an account)")
        if not isinstance(backend_name, str) or not backend_name.strip():
            raise GatewayNotEnabled("a concrete backend name is required")
        factory = self._service_factory or _DefaultServiceFactory()
        service = factory.make_service()          # last-moment; token via IBM only
        from .ibm_runner import RuntimeGateway

        return RuntimeGateway(service=service, backend_name=backend_name)
