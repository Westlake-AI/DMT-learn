__version__ = '0.0.2'


def parse_version_info(version_str: str, length: int = 4) -> tuple:
    from packaging.version import parse
    version = parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        release.extend(list(version.pre))  # type: ignore
    elif version.is_postrelease:
        release.extend(list(version.post))  # type: ignore
    else:
        release.extend([0, 0])
    return tuple(release)


version_info = tuple(int(x) for x in __version__.split('.')[:3])

__all__ = ['__version__', 'version_info', 'parse_version_info']