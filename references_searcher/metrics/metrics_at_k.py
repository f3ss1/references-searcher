from references_searcher import logger


def recall_at_k(
    predicted_items: list[list],
    true_items: list[list],
    k: int = 1,
) -> float:
    """Calculate the recall@k metric.

    Parameters
    ----------
    predicted_items : list
        The items the model predicts.
    true_items : list
        The items the user bought.
    k : int = 1
        The number of predicted items to consider.

    Returns
    -------
    float
        The resulting metric value.

    """
    if len(predicted_items) == 0:
        raise ValueError("The length of 'predicted_items' is zero!")
    if len(true_items) == 0:
        raise ValueError("The length of 'true_items' is zero!")
    if k <= 0:
        raise ValueError("Parameter 'k' should be positive!")

    result = 0
    for i, (user_predicted_items, user_true_items) in enumerate(zip(predicted_items, true_items, strict=True)):
        if len(user_true_items) == 0:
            if len(user_predicted_items) == 0:
                result += 1
            else:
                logger.debug(f"No predictions detected for entry {i}, giving 0 for this entry!")
            continue

        result += sum(
            [user_predicted_item in user_true_items for user_predicted_item in user_predicted_items[:k]],
        ) / len(user_true_items)

    return result / len(predicted_items)


def precision_at_k(
    predicted_items: list[list],
    true_items: list[list],
    k: int = 1,
    normalize: bool = False,
) -> float:
    """Calculate the precision@k metric.

    Parameters
    ----------
    predicted_items : list
        The items the model predicts.
    true_items : list
        The items the user bought.
    k : int = 1
        The number of predicted items to consider.
    normalize : bool = False
        If normalize is set to True, should the number of predicted items be less than k, the denominator will
        be adjusted to the number of predicted items, rather than k. Otherwise, k is used.

    Returns
    -------
    float
        The resulting metric value.

    """
    if len(predicted_items) == 0:
        raise ValueError("The length of 'predicted_items' is zero!")
    if len(true_items) == 0:
        raise ValueError("The length of 'true_items' is zero!")
    if k <= 0:
        raise ValueError("Parameter 'k' should be positive!")

    result = 0
    for i, (user_predicted_items, user_true_items) in enumerate(zip(predicted_items, true_items, strict=True)):
        if len(user_predicted_items) == 0:
            if len(user_true_items) == 0:
                result += 1
            else:
                logger.debug(f"No predictions detected for entry {i}, giving 0 for this entry!")
                continue

        denominator = k
        if len(user_predicted_items) < k and normalize:
            if normalize:
                denominator = len(user_predicted_items)
                logger.debug(f"Normalizing denominator for predictions at entry {i} for {len(user_predicted_items)}")
            else:
                logger.debug(f"Encountered less than {k} predictions at entry {i}: {len(user_predicted_items)}")

        result += (
            sum(
                [user_predicted_item in user_true_items for user_predicted_item in user_predicted_items[:k]],
            )
            / denominator
        )

    return result / len(predicted_items)
