from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from agent import Agent


def test_normalize_actions_prefixes_pyautogui_calls():
    agent = Agent()
    actions = agent._normalize_actions(["click(100, 200)", "press('enter')"])
    assert actions == ["pyautogui.click(100, 200)", "pyautogui.press('enter')"]


def test_normalize_actions_keeps_markers_and_existing_prefixes():
    agent = Agent()
    actions = agent._normalize_actions([
        "wait",
        "DONE",
        "fail",
        "pyautogui.click(1, 2)",
    ])
    assert actions == ["WAIT", "DONE", "FAIL", "pyautogui.click(1, 2)"]


def test_normalize_actions_keeps_unknown_functions_unchanged():
    agent = Agent()
    actions = agent._normalize_actions(["browser.click(10, 20)", "custom_action()"])
    assert actions == ["browser.click(10, 20)", "custom_action()"]


def test_normalize_actions_accepts_type_marker_objects():
    agent = Agent()
    actions = agent._normalize_actions([{"type": "WAIT"}, {"type": "terminate", "status": "success"}])
    assert actions == ["WAIT", "DONE"]


def test_normalize_actions_accepts_action_type_markers():
    agent = Agent()
    actions = agent._normalize_actions([{"action_type": "DONE"}, {"action_type": "FAIL"}])
    assert actions == ["DONE", "FAIL"]


def test_normalize_actions_supports_osworld_computer_use_click_variants():
    agent = Agent()
    actions = agent._normalize_actions([
        {"action": "left_click"},
        {"action": "left_click", "coordinate": [10, 20]},
        {"action": "right_click"},
        {"action": "double_click", "coordinate": [30, 40]},
        {"action": "triple_click", "coordinate": [50, 60]},
    ])
    assert actions == [
        "pyautogui.click()",
        "pyautogui.click(10, 20)",
        "pyautogui.rightClick()",
        "pyautogui.doubleClick(30, 40)",
        "pyautogui.tripleClick(50, 60)",
    ]
