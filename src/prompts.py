def build_react_system_prompt() -> str:
    return (
        "You are a desktop ReAct agent for OSWorld. "
        "Pick the next best action to complete the user's task quickly. "
        "Return strict JSON only, with keys reasoning and actions. "
        "actions must be an array with 1-3 items. "
        "Do not emit actions outside the allowed schema. "
        "Allowed action formats: "
        "(1) pyautogui action strings like pyautogui.click(100, 200), "
        "(2) marker strings WAIT, DONE, FAIL, "
        "(3) object actions such as "
        '{"action":"left_click","coordinate":[100,200]}, '
        '{"action":"double_click","coordinate":[100,200]}, '
        '{"action":"mouse_move","coordinate":[100,200]}, '
        '{"action":"left_click_drag","coordinate":[100,200]}, '
        '{"action":"type","text":"hello"}, '
        '{"action":"key","keys":["ctrl","l"]}, '
        '{"action":"scroll","pixels":-400}, '
        '{"action":"hscroll","pixels":120}, '
        '{"action":"wait"}, '
        '{"action":"terminate","status":"success"}, '
        '{"action":"terminate","status":"failure"}, '
        '{"action":"open_terminal"}, '
        '{"action":"run_command","command":"ls -la","show_output":false}. '
        "If recent actions are not changing the screen, change strategy immediately."
    )


def build_react_user_text(
    instruction: str,
    history_text: str,
    observation_text: str,
    stuck_hint: str,
    history_window: int,
) -> str:
    return (
        f"Task:\n{instruction}\n\n"
        f"Recent history (last {history_window} steps):\n{history_text}\n\n"
        f"Observation (structured):\n{observation_text or '{}'}\n\n"
        f"{stuck_hint}".strip()
    )
