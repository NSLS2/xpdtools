"""PandA settings provider for XPD beamline flyscans."""

from importlib.resources import files
from pathlib import Path
from typing import Any

from ophyd_async.core import YamlSettingsProvider


class PackagedSettingsProvider(YamlSettingsProvider):
    """A read-only YamlSettingsProvider backed by configs shipped with this package.

    Subclasses :class:`ophyd_async.core.YamlSettingsProvider` and points at the
    PandA configuration files bundled in the ``xpdtools.panda_configurations``
    package directory.

    Examples
    --------
    >>> provider = PackagedSettingsProvider()
    >>> provider.available_configs()
    ['single_axis_flyscan']
    >>> import asyncio
    >>> asyncio.run(provider.retrieve("single_axis_flyscan"))  # doctest: +SKIP
    {...}
    """

    def __init__(self) -> None:
        directory = Path(str(files("xpdtools.panda_configurations")))
        super().__init__(directory)

    def available_configs(self) -> list[str]:
        """List available PandA configuration names (without .yaml extension)."""
        return sorted(
            p.stem for p in self._directory.glob("*.yaml") if p.name != "__init__.yaml"
        )

    async def store(self, name: str, data: dict[str, Any]) -> None:
        """Not supported for packaged configs.

        Raises
        ------
        NotImplementedError
            Always, since packaged configs are read-only.
        """
        raise NotImplementedError(
            "Cannot store settings in a packaged provider. "
            "Use ophyd_async.core.YamlSettingsProvider for writable configs."
        )
