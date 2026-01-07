import asyncio
from typing import Any, Awaitable, Callable, List, Optional

from tqdm.auto import tqdm


async def train_with_progress(
    ps: Any,
    rounds: int,
    eval_callback: Optional[Callable[[int], Awaitable[Any]]] = None,
    eval_interval: int = 1,
) -> List[dict]:
    """
    Wraps ParameterServer rounds with a tqdm progress bar and optional evaluation.
    """
    history: List[dict] = []

    with tqdm(total=rounds, desc="Training") as pbar:
        for r in range(1, rounds + 1):
            await ps.round()

            if eval_callback is not None and r % eval_interval == 0:
                pbar.set_description(f"Training (round {r}/{rounds}) - Evaluating")

                metrics = await eval_callback(r)
                history.append({"round": r, "metrics": metrics})

                if isinstance(metrics, dict):
                    pbar.set_postfix(metrics)

            pbar.update(1)
            pbar.set_description(f"Training (round {r}/{rounds})")

    return history
