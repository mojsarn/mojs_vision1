# mcp-vision

An MCP server that exposes a local vision model as callable tools, so a text
agent (opencode, Claude Code, any MCP client) can look at screenshots
mid-reasoning instead of you pasting images in by hand.

The agent calls a tool → this process sends the image to a vision model over
an OpenAI-compatible endpoint → the answer comes back as text.

Built against a llama-swap server running **OS-Atlas-Base-7B** (a GUI
grounding model), but any vision model on an OpenAI-compatible endpoint works.

## Tools

| tool | purpose |
|---|---|
| `take_screenshot(path, region)` | capture the screen (Windows/macOS; needs a display) |
| `look_at_image(path, question)` | open-ended visual question answering |
| `find_ui_element(path, target)` | GUI grounding → pixel click point |
| `list_vision_models()` | what the endpoint offers |

`find_ui_element` returns e.g.

    click_x=1056 click_y=712 box=[973,688,1139,736] image=1280x800

## Why grounding is done inside the tool

OS-Atlas answers with a **normalised** `[x1,y1,x2,y2]` box in 0..1, which is
meaningless until scaled by the image size. The tool always asks for a box and
scales it, because that is the only prompt shape where the model is accurate:

| prompt | result on a 1280x800 test image |
|---|---|
| freeform *"reply with the x,y position"* | `(800, 890)` — **outside the image** |
| bounding box (what the tool sends) | centre `(1056, 712)` vs true `(1060, 722)` |

4 px and 10 px off. Judged on the freeform prompt you would conclude the model
cannot ground at all.

Scaling is done against the **original** image, not the downscaled copy that
gets sent, so the coordinates are true screen pixels ready to click.

## Configuration

| env var | default | meaning |
|---|---|---|
| `LLAMASWAP_URL` | `http://mojs-ai.local:8080/v1` | OpenAI-compatible endpoint |
| `VISION_MODEL` | `os-atlas-base-7b-q6-k` | model id or alias to call |

`max_tokens` is fixed at 900. These are reasoning models: below roughly 600
they spend the whole budget thinking and return an **empty** string.

## Install

### Linux / macOS

    git clone <this repo> ~/mcp-vision && cd ~/mcp-vision
    python3 -m venv venv
    venv/bin/pip install -r requirements.txt
    venv/bin/python mcp-vision.py --selftest some-screenshot.png

### Windows

    py -m venv venv
    venv\Scripts\pip install -r requirements.txt
    venv\Scripts\python mcp-vision.py --selftest some-screenshot.png

`take_screenshot` only works where there is a real display — it uses PIL's
`ImageGrab`, so Windows and macOS yes, headless Linux no.

## Registering with opencode

`opencode.json` (`~/.config/opencode/opencode.json`, or
`%USERPROFILE%\.config\opencode\opencode.json` on Windows):

```json
{
  "mcp": {
    "local-vision": {
      "type": "local",
      "command": ["/home/mojs/mcp-vision/venv/bin/python",
                  "/home/mojs/mcp-vision/mcp-vision.py"],
      "enabled": true,
      "environment": { "LLAMASWAP_URL": "http://mojs-ai.local:8080/v1" }
    }
  }
}
```

On Windows the command becomes:

```json
["E:\\path\\to\\mcp-vision\\venv\\Scripts\\python.exe",
 "E:\\path\\to\\mcp-vision\\mcp-vision.py"]
```

Verify with `opencode mcp list` — it should report `✓ connected`.

If the endpoint is not reachable by mDNS from Windows, set `LLAMASWAP_URL` to
the IP form instead.

## Troubleshooting

**`MCP error -32000: Connection closed`** tells you nothing useful. Run the
server standalone first — `venv/bin/python mcp-vision.py` — and the real error
appears. That is how the `FastMCP` import failure below was found.

**MCP SDK 2.x removed `FastMCP`.** This uses `mcp.server.MCPServer`. On SDK 1.x
the import was `from mcp.server.fastmcp import FastMCP`.

**Empty replies from the vision model** mean the token budget went entirely on
reasoning. Raise `MAX_TOKENS`.
