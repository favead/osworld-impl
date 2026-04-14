import ast
import json
import re
from typing import Any


PYAUTOGUI_FUNCS = {
    "click",
    "doubleClick",
    "tripleClick",
    "rightClick",
    "middleClick",
    "moveTo",
    "move",
    "dragTo",
    "drag",
    "scroll",
    "hscroll",
    "typewrite",
    "write",
    "press",
    "keyDown",
    "keyUp",
    "hotkey",
}


def normalize_actions(raw_actions: list[Any]) -> list[str]:
    normalized: list[str] = []
    for raw_action in raw_actions:
        if isinstance(raw_action, dict):
            mapped = action_from_object(raw_action)
            if mapped:
                normalized.append(mapped)
            continue

        action = str(raw_action).strip()
        if not action:
            continue

        marker = action.upper()
        if marker in {"WAIT", "DONE", "FAIL"}:
            normalized.append(marker)
            continue
        if marker in {"OPEN_TERMINAL", "OPEN_TERMINAL()"}:
            normalized.append(open_terminal_action())
            continue

        if action.startswith("pyautogui."):
            normalized.append(action)
            continue

        run_cmd = parse_run_command_string(action)
        if run_cmd is not None:
            normalized.append(run_cmd)
            continue

        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(", action)
        if match:
            fn_name = match.group(1)
            if fn_name in PYAUTOGUI_FUNCS:
                normalized.append(f"pyautogui.{action}")
                continue

        normalized.append(action)

    return normalized


def action_from_object(action_obj: dict[str, Any]) -> str | None:
    if "arguments" in action_obj and isinstance(action_obj.get("arguments"), dict):
        nested = dict(action_obj["arguments"])
        if "action" not in nested and isinstance(action_obj.get("name"), str):
            nested["action"] = action_obj.get("name")
        return action_from_object(nested)

    if "args" in action_obj and isinstance(action_obj.get("args"), dict):
        nested = dict(action_obj["args"])
        if "action" not in nested and isinstance(action_obj.get("name"), str):
            nested["action"] = action_obj.get("name")
        return action_from_object(nested)

    if action_obj.get("name") == "computer_use" and isinstance(action_obj.get("arguments"), dict):
        return action_from_object(action_obj["arguments"])

    raw_action = action_obj.get("action", action_obj.get("type", action_obj.get("action_type", "")))
    action = str(raw_action).strip().lower()
    if not action:
        return None

    if action in {"done", "success"}:
        return "DONE"
    if action in {"fail", "failed", "failure"}:
        return "FAIL"
    if action == "wait":
        return "WAIT"
    if action == "terminate":
        status = str(action_obj.get("status", "failure")).strip().lower()
        return "DONE" if status == "success" else "FAIL"

    def coord() -> tuple[int, int] | None:
        value = action_obj.get("coordinate")
        if not isinstance(value, list) or len(value) != 2:
            return None
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None

    if action in {"left_click", "click"}:
        point = coord()
        if point:
            x, y = point
            return f"pyautogui.click({x}, {y})"
        return "pyautogui.click()"

    if action == "right_click":
        point = coord()
        if point:
            x, y = point
            return f"pyautogui.rightClick({x}, {y})"
        return "pyautogui.rightClick()"

    if action == "middle_click":
        point = coord()
        if point:
            x, y = point
            return f"pyautogui.middleClick({x}, {y})"
        return "pyautogui.middleClick()"

    if action in {"double_click", "triple_click", "mouse_move", "left_click_drag"}:
        point = coord()
        if not point:
            return None
        x, y = point
        mapping = {
            "double_click": "doubleClick",
            "triple_click": "tripleClick",
            "mouse_move": "moveTo",
            "left_click_drag": "dragTo",
        }
        return f"pyautogui.{mapping[action]}({x}, {y})"

    if action == "type":
        text = action_obj.get("text")
        if isinstance(text, str) and text:
            return f"pyautogui.write({json.dumps(text)})"
        return None

    if action == "key":
        keys = action_obj.get("keys")
        if isinstance(keys, list) and keys:
            cleaned = [str(key).strip() for key in keys if str(key).strip()]
            if not cleaned:
                return None
            if len(cleaned) == 1:
                return f"pyautogui.press({json.dumps(cleaned[0])})"
            args = ", ".join(json.dumps(key) for key in cleaned)
            return f"pyautogui.hotkey({args})"
        return None

    if action in {"scroll", "hscroll"}:
        pixels = action_obj.get("pixels", action_obj.get("clicks"))
        try:
            amount = int(float(pixels))
        except (TypeError, ValueError):
            return None
        if action == "hscroll":
            return f"pyautogui.hscroll({amount})"
        return f"pyautogui.scroll({amount})"

    if action == "open_terminal":
        return open_terminal_action()

    if action == "run_command":
        command = str(action_obj.get("command", "")).strip()
        if not command:
            return None
        show_output = bool(action_obj.get("show_output", False))
        return run_command_action(command, show_output)

    return None


def open_terminal_action() -> str:
    return "import subprocess; subprocess.Popen(['gnome-terminal']); import time; time.sleep(1.5)"


def run_command_action(command: str, show_output: bool) -> str:
    if show_output:
        return (
            "import hashlib as _h, subprocess; "
            f"_cmd={repr(command)}; _tag=_h.md5(_cmd.encode()).hexdigest()[:8]; "
            "_out='/tmp/osworld_cmd_' + _tag + '.txt'; "
            "_r=subprocess.run(_cmd, shell=True, capture_output=True, text=True, timeout=30); "
            "open(_out, 'w').write('$ ' + _cmd + '\\n' + (_r.stdout or '') + (_r.stderr or '')); "
            "subprocess.Popen(['gnome-terminal', '--', 'bash', '-lc', 'cat ' + _out + '; exec bash'])"
        )
    return f"import subprocess; subprocess.Popen({repr(command)}, shell=True)"


def parse_run_command_string(action: str) -> str | None:
    if not action.startswith("run_command(") or not action.endswith(")"):
        return None

    args_src = action[len("run_command(") : -1].strip()
    if not args_src:
        return None

    show_output = False
    command = ""
    try:
        if "," in args_src:
            left, right = args_src.rsplit(",", 1)
            if "show_output" in right:
                args_src = left.strip()
                _, _, value = right.partition("=")
                show_output = str(value).strip().lower() in {"1", "true", "yes"}
        command = ast.literal_eval(args_src)
    except Exception:
        return None

    if not isinstance(command, str) or not command.strip():
        return None
    return run_command_action(command.strip(), show_output)


def is_terminal_action(action: str) -> bool:
    return (
        "gnome-terminal" in action
        or "subprocess.Popen(" in action
        or action.startswith("run_command(")
        or action.upper() in {"OPEN_TERMINAL", "OPEN_TERMINAL()"}
    )


def has_terminal_action(actions: list[str]) -> bool:
    return any(is_terminal_action(action) for action in actions)
