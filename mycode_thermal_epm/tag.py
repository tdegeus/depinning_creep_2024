import packaging.version


def has_uncommitted(ver: str):
    """
    Check of a version string encoded that there were uncommitted changes.

    :param a: Version string.
    :return: ``True`` is there were uncommitted changes.
    """

    V = packaging.version.parse(ver)

    if not V.local:
        return False

    return len(V.local.split(".")) > 1


def any_has_uncommitted(deps: list[str]) -> bool:
    """
    Check if any of the dependencies in ``deps`` has an uncommitted change.

    :param deps: List of versions formatted as ["name=versionstring", ...].
    :return: ``True`` if any of the dependencies has an uncommitted change.
    """

    V = {lib.split("=")[0]: lib.split("=")[1] for lib in deps}

    for lib in V:
        if has_uncommitted(V[lib]):
            return True

    return False


def less(a: str, b: str) -> bool:
    """
    Check if ``a`` is a version less than that in ``b``.

    :param a: Version string.
    :param b: Version string.
    :return: ``True`` if ``a < b``
    """

    return packaging.version.parse(a) < packaging.version.parse(b)


def less_equal(a: str, b: str) -> bool:
    """
    Check if ``a`` is a version less or equal than that in ``b``.

    :param a: Version string.
    :param b: Version string.
    :return: ``True`` if ``a <= b``
    """

    return packaging.version.parse(a) <= packaging.version.parse(b)


def greater(a: str, b: str) -> bool:
    """
    Check if ``a`` is a version greater than that in ``b``.

    :param a: Version string.
    :param b: Version string.
    :return: ``True`` if ``a > b``
    """

    return packaging.version.parse(a) > packaging.version.parse(b)


def greater_equal(a: str, b: str) -> bool:
    """
    Check if ``a`` is a version greater or equal than that in ``b``.

    :param a: Version string.
    :param b: Version string.
    :return: ``True`` if ``a >= b``
    """

    return packaging.version.parse(a) >= packaging.version.parse(b)


def equal(a: str, b: str) -> bool:
    """
    Check if version ``a`` is equal to ``b``.

    :param a: Version string.
    :param b: Version string.
    :return: ``True`` if ``a == b``
    """

    return packaging.version.parse(a) == packaging.version.parse(b)


def all_equal(a: list[str], b: list[str]) -> bool:
    """
    Check if all dependencies in ``a`` have a version equal to those in ``b``.
    It is allowed to have dependencies in ``a`` or ``b`` that are not in the other list.

    :param a: List of versions formatted as ["name=versionstring", ...].
    :param b: List of versions formatted as ["name=versionstring", ...].
    :return: ``True`` if all dependencies in ``a`` are equal to those in ``b``
    """

    A = {deps.split("=")[0]: deps.split("=")[1] for deps in a}
    B = {deps.split("=")[0]: deps.split("=")[1] for deps in b}

    for lib in A:
        if lib not in B:
            continue
        if packaging.version.parse(A[lib]) == packaging.version.parse(B[lib]):
            continue
        return False

    return True


def all_greater_equal(a: list[str], b: list[str]) -> bool:
    """
    Check if all dependencies in ``a`` have a version greater or equal than those in ``b``.
    It is allowed to have dependencies in ``a`` or ``b`` that are not in the other list.

    :param a: List of versions formatted as ["name=versionstring", ...].
    :param b: List of versions formatted as ["name=versionstring", ...].
    :return: ``True`` if all dependencies in ``a`` are greater or equal than those in ``b``
    """

    A = {deps.split("=")[0]: deps.split("=")[1] for deps in a}
    B = {deps.split("=")[0]: deps.split("=")[1] for deps in b}

    for lib in A:
        if lib not in B:
            continue
        if packaging.version.parse(A[lib]) >= packaging.version.parse(B[lib]):
            continue
        return False

    return True
