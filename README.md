# mojs_vision1

An MCP server that exposes a local vision model as callable tools, so a text
agent (opencode, Claude Code, any MCP client) can look at the live screen
and interact with UI elements mid-reasoning.

Built against a llama-swap server running **OS-Atlas-Base-7B** (a GUI
grounding model), but any vision model on an OpenAI-compatible endpoint works.

## Tools

| tool | purpose |
|---|---|
| `list_windows()` | list all open windows |
| `ui_inspect(question, window)` | ask a question about what's on screen |
| `find_ui_element(target, window)` | locate a UI element, returns pixel coords |
| `interact_ui_element(target, action, text, window)` | find + click/type/scroll/etc |
| `write_text(text, window)` | type text at cursor (no vision, fast) |

All tools auto-screenshot — no need to capture first.

The `window` parameter is optional. Pass a partial window title (e.g. "Gmail"
or "opencode") to focus on that window before capturing.

### interact_ui_element actions

| action | what it does |
|---|---|
| `click` | single click on the element center |
| `double_click` | double-click on the element center |
| `right_click` | right-click (context menu) |
| `type` | click the element, then paste the text |
| `hover` | move the mouse to the element center |
| `scroll` | scroll down at the element center |
| `key` | press a keyboard key (pass key name in `text`) |
| `hotkey` | press key combination (pass in `text`, e.g. "ctrl+a") |

### write_text

Types text at the current cursor position without using the vision model.
Much faster than `interact_ui_element` with `action='type'`.

Workflow: click the target element first, then call `write_text`.

## How to write effective targets

The vision model locates elements by visual appearance. Target descriptions
should describe what the element **looks like**, not just what it does.

| target | quality |
|---|---|
| `"the button"` | bad — too vague |
| `"the red Submit button at the bottom of the form"` | good |
| `"the maximize restore button in the title bar, next to the X"` | good |
| `"the search input field with a magnifying glass icon"` | good |

`ui_inspect` works best with yes/no or factual questions. It may give
unreliable answers for open-ended descriptions like "describe everything".

## Configuration

| env var | default | meaning |
|---|---|---|
| `LLAMASWAP_URL` | `http://mojs-ai.local:8080/v1` | OpenAI-compatible endpoint |
| `VISION_MODEL` | `os-atlas-base-7b-q6-k` | model id or alias to call |

`max_tokens` is fixed at 900. These are reasoning models: below roughly 600
they spend the whole budget thinking and return an **empty** string.

## Install

### Linux / macOS

    git clone <this repo> ~/mojs_vision1 && cd ~/mojs_vision1
    python3 -m venv venv
    venv/bin/pip install -r requirements.txt
    venv/bin/python mcp-vision.py --selftest

### Windows

    py -m venv venv
    venv\Scripts\pip install -r requirements.txt
    venv\Scripts\python mcp-vision.py --selftest

Screen capture uses PIL's `ImageGrab`, so Windows and macOS yes,
headless Linux no.

## Registering with opencode

`opencode.json` (`~/.config/opencode/opencode.json`, or
`%USERPROFILE%\.config\opencode\opencode.json` on Windows):

```json
{
  "mcp": {
    "vision": {
      "type": "local",
      "command": ["/home/mojs/mojs_vision1/venv/bin/python",
                  "/home/mojs/mojs_vision1/mcp-vision.py"],
      "enabled": true,
      "environment": { "LLAMASWAP_URL": "http://mojs-ai.local:8080/v1" }
    }
  }
}
```

On Windows the command becomes:

```json
["C:\\path\\to\\python.exe",
 "C:\\path\\to\\mcp-vision.py"]
```

Verify with `opencode mcp list` — it should report `✓ connected`.

If the endpoint is not reachable by mDNS from Windows, set `LLAMASWAP_URL` to
the IP form instead.

## Troubleshooting

**`MCP error -32000: Connection closed`** tells you nothing useful. Run the
server standalone first — `venv/bin/python mcp-vision.py` — and the real error
appears.

**Empty replies from the vision model** mean the token budget went entirely on
reasoning. Raise `MAX_TOKENS`.
