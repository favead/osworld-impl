import json


def build_react_system_prompt() -> str:
    description_prompt_lines = [
        "Use a mouse and keyboard to interact with a computer, and take screenshots.",
        "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
        "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.",
        "* The screen's resolution is 1000x1000.",
        "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
        "* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
        "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
    ]
    description_prompt = "\n".join(description_prompt_lines)

    action_description_prompt = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action).
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question.
* `open_terminal`: Open a terminal window.
* `run_command`: Run a terminal command.
    """.strip()

    tools_def = {
        "type": "function",
        "function": {
            "name_for_human": "computer_use",
            "name": "computer_use",
            "description": description_prompt,
            "parameters": {
                "properties": {
                    "action": {
                        "description": action_description_prompt,
                        "enum": [
                            "key",
                            "type",
                            "mouse_move",
                            "left_click",
                            "left_click_drag",
                            "right_click",
                            "middle_click",
                            "double_click",
                            "triple_click",
                            "scroll",
                            "hscroll",
                            "wait",
                            "terminate",
                            "answer",
                            "open_terminal",
                            "run_command",
                        ],
                        "type": "string",
                    },
                    "keys": {"description": "Required only by `action=key`.", "type": "array"},
                    "text": {"description": "Required only by `action=type`.", "type": "string"},
                    "coordinate": {"description": "The x,y coordinates for mouse actions.", "type": "array"},
                    "pixels": {"description": "The amount of scrolling.", "type": "number"},
                    "time": {"description": "The seconds to wait.", "type": "number"},
                    "status": {
                        "description": "The status of the task.",
                        "type": "string",
                        "enum": ["success", "failure"],
                    },
                    "command": {"description": "Required only by `action=run_command`.", "type": "string"},
                    "show_output": {
                        "description": "If true, include command output in observation.",
                        "type": "boolean",
                    },
                },
                "required": ["action"],
                "type": "object",
            },
            "args_format": "Format the arguments as a JSON object.",
        },
    }

    return (
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n"
        f"{json.dumps(tools_def)}\n"
        "</tools>\n\n"
        "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>\n\n"
        "# Few-shot examples\n\n"
        "Example 1:\n"
        "Action: Move the cursor to the Firefox icon and click it.\n"
        "<tool_call>\n"
        '{"name":"computer_use","arguments":{"action":"left_click","coordinate":[110,915]}}\n'
        "</tool_call>\n\n"
        "Example 2:\n"
        "Action: Wait briefly for the application window to appear.\n"
        "<tool_call>\n"
        '{"name":"computer_use","arguments":{"action":"wait","time":2}}\n'
        "</tool_call>\n\n"
        "Example 3:\n"
        "Action: Open a terminal window to run commands.\n"
        "<tool_call>\n"
        '{"name":"computer_use","arguments":{"action":"open_terminal"}}\n'
        "</tool_call>\n\n"
        "Example 4:\n"
        "Action: Run a command to list files in the current directory.\n"
        "<tool_call>\n"
        '{"name":"computer_use","arguments":{"action":"run_command","command":"ls","show_output":true}}\n'
        "</tool_call>\n\n"
        "Example 5:\n"
        "Action: Complete the task and report success.\n"
        "<tool_call>\n"
        '{"name":"computer_use","arguments":{"action":"terminate","status":"success"}}\n'
        "</tool_call>\n\n"
        "# Response format\n\n"
        "Response format for every step:\n"
        "1) Action: a short imperative describing what to do in the UI.\n"
        "2) A single <tool_call>...</tool_call> block containing only the JSON: {\"name\": <function-name>, \"arguments\": <args-json-object>}.\n\n"
        "Rules:\n"
        "- Output exactly in the order: Action, <tool_call>.\n"
        "- Be brief: one sentence for Action.\n"
        "- Do not output anything else outside those parts.\n"
        "- If finishing, use action=terminate in the tool call.\n"
        "- If recent actions are not changing the screen, change strategy immediately."
    )


def build_react_user_text(
    instruction: str,
    history_text: str,
    observation_text: str,
    stuck_hint: str,
    history_window: int,
) -> str:
    parts = [
        "Please generate the next move according to the UI screenshot, instruction and previous actions.",
        "",
        f"Instruction: {instruction}",
        "",
        "Previous actions:",
        history_text,
        "",
        "Observation (structured):",
        observation_text or "{}",
    ]
    if stuck_hint:
        parts.extend(["", stuck_hint])
    parts.extend(["", f"Use the recent history window as {history_window} steps."])
    return "\n".join(parts).strip()
