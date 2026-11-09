import numpy as np


ROOT_LIKE_TYPES = {"ROOT", "STEM"}
LINK_LIKE_TYPES = {"LINK", "INTERFIX"}
NEXT_MORPHEME_TYPE_ORDER = (
    "END",
    "SUFF",
    "LINK",
    "INTERFIX",
    "POST",
    "PREF",
    "ROOT",
    "STEM",
)
INITIAL_MORPHEME_TYPES = {"PREF", "ROOT", "STEM", "None"}
FINAL_MORPHEME_TYPES = {"ROOT", "STEM", "SUFF", "END", "POST", "None"}


def get_next_morpheme_types(morpheme_type: str) -> list[str]:
    """Return morpheme types that can follow the current morpheme type."""
    if morpheme_type == "None":
        return ["None"]

    if morpheme_type in ROOT_LIKE_TYPES | {"SUFF", "HYPN"}:
        next_types = list(NEXT_MORPHEME_TYPE_ORDER)
    elif morpheme_type == "END":
        next_types = list(NEXT_MORPHEME_TYPE_ORDER[1:])
    elif morpheme_type in LINK_LIKE_TYPES | {"PREF", "BEGIN"}:
        next_types = ["PREF", "ROOT", "STEM"]
    else:
        return []

    if next_types and morpheme_type != "HYPN":
        next_types.append("HYPN")
    if morpheme_type == "BEGIN":
        next_types.append("None")
    return next_types


def get_next_morpheme(morpheme: str) -> list[str]:
    label, mtype = morpheme.split("-", 1)
    if label in {"B", "M"}:
        next_labels = ("M", "E")
        next_types = (mtype,)
    else:
        next_labels = ("B", "S")
        next_types = get_next_morpheme_types(mtype)
    return [f"{label}-{mtype}" for label in next_labels for mtype in next_types]


def is_correct_morpheme_sequence(morphemes: list[str]) -> bool:
    """Check whether labels form a valid BMES morpheme sequence."""
    if not morphemes or any("-" not in morpheme for morpheme in morphemes):
        return False

    first_label, first_type = morphemes[0].split("-", 1)
    if first_label not in {"B", "S"} or first_type not in INITIAL_MORPHEME_TYPES:
        return False

    last_label, last_type = morphemes[-1].split("-", 1)
    if last_label not in {"E", "S"} or last_type not in FINAL_MORPHEME_TYPES:
        return False

    for current, next_morpheme in zip(morphemes, morphemes[1:]):
        if next_morpheme not in get_next_morpheme(current):
            return False
    return True


def get_initial_label_ids(label_list: list[str]) -> list[int]:
    return [
        i
        for i, label in enumerate(label_list)
        if "-" in label
        and label.split("-", 1)[0] in {"B", "S"}
        and label.split("-", 1)[1] in INITIAL_MORPHEME_TYPES
    ]


def get_final_label_ids(label_list: list[str]) -> list[int]:
    return [
        i
        for i, label in enumerate(label_list)
        if "-" in label
        and label.split("-", 1)[0] in {"E", "S"}
        and label.split("-", 1)[1] in FINAL_MORPHEME_TYPES
    ]


def get_next_label_ids(
    label_id: int,
    label_list: list[str],
    label_to_id: dict[str, int],
) -> list[int]:
    label = label_list[label_id]
    if "-" not in label:
        return []
    return [
        label_to_id[next_label]
        for next_label in get_next_morpheme(label)
        if next_label in label_to_id
    ]


def decode_valid_bmes(
    probs: np.ndarray,
    label_list: list[str],
    length: int | None = None,
    label_to_id: dict[str, int] | None = None,
) -> list[str]:
    """
    Choose the most probable BMES sequence allowed by morpheme-order rules.

    probs must contain probabilities for real word characters only; BEGIN is a
    virtual state and must not be included into label_list.
    """
    if length is None:
        length = len(probs)
    if length == 0:
        return []

    label_to_id = label_to_id or {label: i for i, label in enumerate(label_list)}
    probs = np.asarray(probs[:length], dtype=np.float64)

    greedy_ids = np.argmax(probs, axis=1)
    greedy_labels = [label_list[i] for i in greedy_ids]
    if is_correct_morpheme_sequence(greedy_labels):
        return greedy_labels

    n_labels = len(label_list)
    safe_probs = np.clip(probs, 1e-45, 1.0)
    costs = np.full((length, n_labels), np.inf, dtype=np.float64)
    backpointers = np.full((length, n_labels), -1, dtype=np.int64)

    for label_id in get_initial_label_ids(label_list):
        costs[0, label_id] = -np.log(safe_probs[0, label_id])

    transitions = {
        label_id: get_next_label_ids(label_id, label_list, label_to_id)
        for label_id in range(n_labels)
    }

    for pos in range(1, length):
        reachable_prev_ids = np.where(np.isfinite(costs[pos - 1]))[0]
        for prev_id in reachable_prev_ids:
            prev_cost = costs[pos - 1, prev_id]
            for next_id in transitions[prev_id]:
                candidate_cost = prev_cost - np.log(safe_probs[pos, next_id])
                if candidate_cost < costs[pos, next_id]:
                    costs[pos, next_id] = candidate_cost
                    backpointers[pos, next_id] = prev_id

    final_ids = [
        label_id
        for label_id in get_final_label_ids(label_list)
        if np.isfinite(costs[-1, label_id])
    ]
    if not final_ids:
        return greedy_labels

    best_id = min(final_ids, key=lambda label_id: costs[-1, label_id])
    best_path = [best_id]
    for pos in range(length - 1, 0, -1):
        best_id = backpointers[pos, best_id]
        if best_id == -1:
            return greedy_labels
        best_path.append(best_id)

    best_path.reverse()
    return [label_list[i] for i in best_path]


def decode_valid_bmes_topk(
    probs: np.ndarray,
    label_list: list[str],
    k: int,
    length: int | None = None,
    label_to_id: dict[str, int] | None = None,
    *,
    log_scores: list[float] | None = None,
) -> list[list[str]]:
    """Return up to k valid BMES paths, ordered by decreasing probability.

    Keep k paths per state (exact k-best Viterbi, not a global beam).
    Use the same probability clipping and morpheme-order rules as top-1.
    Return [] if no valid complete path exists; never fall back to argmax.
    If log_scores is provided, append the log score of each returned path.
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    if length is None:
        length = len(probs)
    if not 0 <= length <= len(probs):
        raise ValueError("length must be between 0 and len(probs).")
    if length == 0:
        return []

    label_to_id = label_to_id or {label: i for i, label in enumerate(label_list)}
    probs = np.asarray(probs[:length], dtype=np.float64)
    emission_costs = -np.log(np.clip(probs, 1e-45, 1.0))
    n_labels = len(label_list)
    costs = np.full((n_labels, k), np.inf, dtype=np.float64)
    # Each backpointer stores both the previous state and its path rank.
    back_states = np.full((length, n_labels, k), -1, dtype=np.int64)
    back_ranks = np.full_like(back_states, -1)
    for label_id in get_initial_label_ids(label_list):
        costs[label_id, 0] = emission_costs[0, label_id]

    predecessors = [[] for _ in label_list]
    for prev_id in range(n_labels):
        for next_id in get_next_label_ids(prev_id, label_list, label_to_id):
            predecessors[next_id].append(prev_id)

    for pos in range(1, length):
        next_costs = np.full_like(costs, np.inf)
        for next_id, prev_ids in enumerate(predecessors):
            if not prev_ids:
                continue
            candidate_costs = costs[prev_ids].reshape(-1) + emission_costs[pos, next_id]
            # Stable ties: predecessor label ID, then predecessor path rank.
            best_indices = np.argsort(candidate_costs, kind="stable")[:k]
            for rank, index in enumerate(best_indices):
                if not np.isfinite(candidate_costs[index]):
                    break
                next_costs[next_id, rank] = candidate_costs[index]
                back_states[pos, next_id, rank] = prev_ids[index // k]
                back_ranks[pos, next_id, rank] = index % k
        costs = next_costs

    final_ids = get_final_label_ids(label_list)
    final_costs = costs[final_ids].reshape(-1)
    paths = []
    for index in np.argsort(final_costs, kind="stable")[:k]:
        if not np.isfinite(final_costs[index]):
            break
        state, rank = final_ids[index // k], index % k
        path = [state]
        for pos in range(length - 1, 0, -1):
            state, rank = back_states[pos, state, rank], back_ranks[pos, state, rank]
            path.append(state)
        paths.append([label_list[i] for i in reversed(path)])
        if log_scores is not None:
            log_scores.append(-float(final_costs[index]))
    return paths
