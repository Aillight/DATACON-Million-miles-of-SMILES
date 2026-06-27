from __future__ import annotations

import time
from pathlib import Path
from typing import Callable


SleepFn = Callable[[float], None]


def safe_unlink_path(
    path: Path,
    attempts: int = 5,
    delay_seconds: float = 0.2,
    sleep: SleepFn = time.sleep,
) -> bool:
    for attempt in range(max(1, attempts)):
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            if attempt == attempts - 1:
                return False
            sleep(delay_seconds)
    return False
