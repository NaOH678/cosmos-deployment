"""Experimental EEF prefix payload; commands stay in the existing executor."""
import time
import numpy as np

METHOD = "identity_jacobian_soft_prefix_v1"

def build_prefix(queue, soft_tail, *, horizon, method=METHOD):
    hard = len(queue)
    if hard + horizon > 32:
        raise ValueError("RTC committed prefix exceeds prediction budget")
    actions = [entry.action for entry in queue] + list(soft_tail[:2])
    values = []
    for action in actions:
        arm = action["arm_action_right"]
        row = np.concatenate((arm["ee_pos"], arm["ee_quat"],
                              np.deg2rad(action["hand_action_right"])))
        if row.shape != (27,) or not np.isfinite(row).all():
            raise ValueError("invalid RTC EEF/hand prefix")
        values.append(row.tolist())
    return dict(method=method, committed_steps=hard, actions=values,
                captured_monotonic=time.monotonic(),
                sequences=[entry.sequence for entry in queue],
                due_at=[entry.due_at for entry in queue])
