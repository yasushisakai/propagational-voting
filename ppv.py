import logging
import warnings
from typing import Literal, NamedTuple

import numpy as np

log = logging.getLogger(__name__)

Role = Literal["delegate", "intermediate"]


class Consensus(NamedTuple):
    label: str
    value: float


class Influence(NamedTuple):
    label: str
    role: Role
    value: float


def compute(
    delegates: dict[str, dict[str, float]],
    intermediates: dict[str, dict[str, float]],
    policies: list[str],
    tol: float = 1e-9,
    max_iter: int = 10_000,
) -> tuple[list[Consensus], list[Influence]]:
    """High level Propagational Proxy Voting from a human-friendly sparse description.

    Each delegate and intermediate is a dict mapping target-name → weight. Weights
    in a single voter's dict must sum to 1.0 (they describe how that voter splits
    their unit of voting power). Policies are absorbing states; you only pass
    their names — the function inserts the identity block for you.

    Ordering inside the matrix is fixed: delegates first, then intermediates,
    then policies. Labels must be unique across all three groups.

    Args:
        delegates: Mapping of delegate name to their outgoing votes,
            e.g. ``{'Alice': {'Bob': 0.2, 'FAR2': 0.8}}``. Targets may be other
            delegates, intermediates, or policies. A voter cannot vote for
            themselves (no self-key).
        intermediates: Same shape as ``delegates``. Intermediates re-distribute
            mass they receive but are not themselves a final destination.
        policies: Ordered list of policy names (absorbing states). The order here
            controls the policy row/column order in the underlying matrix.
        tol: Convergence threshold on the max remaining transient mass. Stops
            iterating once ``A_k[:ndi, :].max() < tol``.
        max_iter: Hard cap on iterations. If exceeded without convergence, emits
            a ``UserWarning`` but still returns the partial result.

    Returns:
        ``(consensus, influences)`` where:
          - ``consensus``: list of :class:`Consensus`, sorted descending by value,
            one entry per policy.
          - ``influences``: list of :class:`Influence`, sorted descending by value,
            one entry per delegate + intermediate.

    Example:
        >>> delegates = {
        ...     'Alice': {'RedFruit': 0.3, 'apple': 0.7},
        ...     'Bob':   {'Alice': 0.2, 'banana': 0.8},
        ... }
        >>> intermediates = {'RedFruit': {'apple': 1.0}}
        >>> policies = ['apple', 'banana']
        >>> consensus, influences = compute(delegates, intermediates, policies)
        >>> consensus[0].label  # winning policy
        'apple'

    Raises:
        AssertionError: If labels are not unique, columns don't sum to 1, a voter
            votes for themselves, or policies aren't strictly absorbing.

    Debug:
        The assembled voting matrix is emitted at ``logging.DEBUG`` on the
        ``"ppv"`` logger. Enable with::

            import logging
            logging.basicConfig(level=logging.DEBUG)

    See Also:
        :func:`build_matrix` for the dict→matrix step on its own.
        :func:`compute_matrix` for the propagation kernel.
    """
    v, labels, num_delegates, num_intermediates = build_matrix(
        delegates, intermediates, policies
    )

    if log.isEnabledFor(logging.DEBUG):
        with np.printoptions(precision=3, suppress=True, linewidth=120):
            log.debug("voting matrix (labels=%s):\n%s", labels, v)

    return compute_matrix(
        v, labels, num_delegates, num_intermediates, tol=tol, max_iter=max_iter
    )


def build_matrix(
    delegates: dict[str, dict[str, float]],
    intermediates: dict[str, dict[str, float]],
    policies: list[str],
) -> tuple[np.ndarray, list[str], int, int]:
    """Assemble the column-stochastic voting matrix from sparse inputs.

    This is the matrix-construction half of :func:`compute`, exposed so callers
    can inspect or modify ``v`` before handing it to :func:`compute_matrix`
    (debugging, perturbation analysis, caching, etc.).

    Args:
        delegates: See :func:`compute`.
        intermediates: See :func:`compute`.
        policies: See :func:`compute`.

    Returns:
        ``(v, labels, num_delegates, num_intermediates)`` — exactly the four
        positional arguments :func:`compute_matrix` expects. Labels are ordered
        ``list(delegates) + list(intermediates) + list(policies)``.

    Example:
        >>> v, labels, nd, ni = build_matrix(
        ...     delegates={'Alice': {'apple': 1.0}, 'Bob': {'banana': 1.0}},
        ...     intermediates={'RedFruit': {'apple': 1.0}},
        ...     policies=['apple', 'banana'],
        ... )
        >>> labels
        ['Alice', 'Bob', 'RedFruit', 'apple', 'banana']
        >>> nd, ni
        (2, 1)
    """
    labels = list(delegates) + list(intermediates) + list(policies)
    assert len(labels) == len(set(labels)), (
        "labels must be unique across delegates, intermediates, policies"
    )

    index_of = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    num_delegates = len(delegates)
    num_intermediates = len(intermediates)

    v = np.zeros((n, n))
    for voter, edges in (delegates | intermediates).items():
        j = index_of[voter]
        for target, weight in edges.items():
            v[index_of[target], j] = weight
    for p in policies:
        i = index_of[p]
        v[i, i] = 1.0

    return v, labels, num_delegates, num_intermediates


def compute_matrix(
    v: np.ndarray,
    labels: list[str],
    num_delegates: int,
    num_intermediates: int,
    tol: float = 1e-9,
    max_iter: int = 10_000,
) -> tuple[list[Consensus], list[Influence]]:
    """Run Propagational Proxy Voting from a column-stochastic matrix.

    This is the low-level kernel. Use it when you already have the voting
    matrix as a numpy array. The matrix layout is positional and strict:

    - Column ``j`` represents voter ``j``'s outgoing vote distribution and
      must sum to 1 (column-stochastic).
    - Rows/columns ``[0, num_delegates)`` are delegates.
    - Rows/columns ``[num_delegates, num_delegates + num_intermediates)`` are
      intermediates.
    - The remaining rows/columns are policies. The policy × policy block must
      be the identity matrix (absorbing states), and policies must have zero
      outgoing votes to non-policy rows.

    Args:
        v: Square ``(n, n)`` column-stochastic ndarray. ``v[i, j]`` is the
            fraction of voter ``j``'s voting power that flows to entity ``i``.
        labels: Length-``n`` list of names, in the same order as the matrix
            rows/columns. Used to label the returned tuples.
        num_delegates: Number of delegate rows/columns at the top of ``v``.
        num_intermediates: Number of intermediate rows/columns immediately
            after the delegates.
        tol: Convergence threshold on the max remaining transient mass. Stops
            iterating once ``A_k[:ndi, :].max() < tol``.
        max_iter: Hard cap on iterations. If exceeded without convergence,
            emits a ``UserWarning`` but still returns the partial result.

    Returns:
        See :func:`compute`

    Example:

        >>> import numpy as np
        >>> v = np.array([
        ...     #  Alice  Bob  RedFruit  apple  banana
        ...     [   0.0,  0.2,    0.0,    0.0,   0.0 ],  # → Alice
        ...     [   0.0,  0.0,    0.0,    0.0,   0.0 ],  # → Bob
        ...     [   0.3,  0.0,    0.0,    0.0,   0.0 ],  # → RedFruit
        ...     [   0.7,  0.0,    1.0,    1.0,   0.0 ],  # → apple (absorbing)
        ...     [   0.0,  0.8,    0.0,    0.0,   1.0 ],  # → banana (absorbing)
        ... ])
        >>> consensus, influences = compute_matrix(
        ...     v,
        ...     labels=['Alice', 'Bob', 'RedFruit', 'apple', 'banana'],
        ...     num_delegates=2,
        ...     num_intermediates=1,
        ... )
        >>> consensus[0].label
        'apple'

    Raises:
        AssertionError: If ``v`` isn't square, labels don't match its size,
            columns don't sum to 1, voters self-vote, or the policy block
            isn't a clean absorbing identity.
    """
    n = v.shape[0]
    ndi = num_delegates + num_intermediates
    num_policies = n - ndi

    assert v.shape == (n, n), "v must be square"
    assert len(labels) == n, "labels must match matrix size"
    assert np.allclose(v.sum(axis=0), 1.0), "columns must sum to 1 (column-stochastic)"
    assert np.allclose(np.diag(v[:ndi, :ndi]), 0), "voters cannot vote for themselves"
    assert np.allclose(v[ndi:, ndi:], np.eye(num_policies)), (
        "policies must be absorbing (identity block)"
    )
    assert np.allclose(v[:ndi, ndi:], 0), "policies must not vote outward"

    a = np.eye(n)
    influence = a.copy()

    for _ in range(max_iter):
        a = a @ v
        influence += a
        if a[:ndi, :].max() < tol:
            break
    else:
        warnings.warn(
            f"did not converge within {max_iter} iterations "
            f"(max transient mass = {a[:ndi, :].max():.2e})",
            stacklevel=2,
        )

    policy_totals = a[ndi:, :num_delegates].sum(axis=1).tolist()
    consensus = sorted(
        (
            Consensus(label, value)
            for label, value in zip(labels[ndi:], policy_totals, strict=True)
        ),
        key=lambda c: c.value,
        reverse=True,
    )

    block = influence[:ndi, :ndi]
    inf_values = (block.sum(axis=1) / block.diagonal()).tolist()
    roles: list[Role] = ["delegate"] * num_delegates + [
        "intermediate"
    ] * num_intermediates
    influences = sorted(
        (
            Influence(label, role, value)
            for label, role, value in zip(labels[:ndi], roles, inf_values, strict=True)
        ),
        key=lambda i: i.value,
        reverse=True,
    )

    return (consensus, influences)
