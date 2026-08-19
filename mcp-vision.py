#!/usr/bin/env python3
"""MCP server exposing the local vision model as callable tools.

Lets a text agent (opencode, Claude Code, anything MCP-capable) look at
screenshots mid-reasoning instead of you pasting images by hand. The
agent calls a tool, this process asks OS-Atlas over the local llama-swap
endpoint, and returns the answer.

Tools:
  take_screenshot(path, region)     grab the screen (Windows/macOS)
  look_at_image(path, question)     describe / answer about an image
  find_ui_element(path, target)     GUI grounding -> pixel click point
  list_vision_models()              what's available to point --model at

Run directly for a smoke test:
  tools/venv/bin/python tools/mcp-vision.py --selftest <image>
"""
import argparse, ast, base64, io, os, re, sys

API_BASE = os.environ.get("LLAMASWAP_URL", "http://mojs-ai.local:8080/v1")
VISION_MODEL = os.environ.get("VISION_MODEL", "os-atlas-base-7b-q6-k")
MAX_TOKENS = 900          # reasoning models return EMPTY below ~600 (CLAUDE.md)
MAX_PX = 1600             # full-res screenshots cost tokens for no accuracy gain


def _client():
    from openai import OpenAI
    return OpenAI(base_url=API_BASE, api_key="local")


def _encode(path, max_px=MAX_PX):
    from PIL import Image
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such image: {path}")
    im = Image.open(path)
    orig = im.size
    if max(im.size) > max_px:
        s = max_px / max(im.size)
        im = im.resize((int(im.size[0] * s), int(im.size[1] * s)))
    if im.mode != "RGB":
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), im.size, orig


def _ask_vision(path, prompt, model=None):
    b64, size, orig = _encode(path)
    r = _client().chat.completions.create(
        model=model or VISION_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64," + b64}}]}],
        max_tokens=MAX_TOKENS, temperature=0)
    txt = (r.choices[0].message.content or "").strip()
    if not txt:
        raise RuntimeError("vision model returned empty content "
                           "(reasoning consumed the token budget)")
    return txt, size, orig


def look_at_image(path: str, question: str = "Describe this screenshot.",
                  model: str = "") -> str:
    """Ask the vision model an open question about an image."""
    txt, size, orig = _ask_vision(path, question, model or None)
    return f"{txt}\n\n(image {orig[0]}x{orig[1]}, sent as {size[0]}x{size[1]})"


def find_ui_element(path: str, target: str, model: str = "") -> str:
    """Locate a UI element and return a pixel click point.

    OS-Atlas is a GUI-grounding model: asked for a bounding box it answers
    with NORMALISED 0-1 coordinates and is accurate to a few pixels. Asked
    freeform for "x,y" it returns coordinates outside the image. So always
    request a box, then scale it here.
    """
    prompt = (f'In this UI screenshot, what is the bounding box of the element '
              f'corresponding to the command "{target}"? Reply with coordinates only.')
    txt, size, orig = _ask_vision(path, prompt, model or None)
    m = re.search(r"\[[^\]]+\]", txt)
    if not m:
        return f"No bounding box in the model's reply: {txt[:200]}"
    try:
        box = [float(v) for v in ast.literal_eval(m.group(0))]
    except (ValueError, SyntaxError):
        return f"Unparseable box {m.group(0)!r} (raw reply: {txt[:200]})"
    if len(box) != 4:
        return f"Expected 4 coordinates, got {box} (raw: {txt[:200]})"
    # scale against the ORIGINAL image, so the caller gets true screen pixels
    w, h = orig
    if max(box) <= 1.0:
        x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    else:                                    # already pixels, in sent-size space
        sx, sy = w / size[0], h / size[1]
        x1, y1, x2, y2 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return (f"click_x={cx:.0f} click_y={cy:.0f} "
            f"box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] "
            f"image={w}x{h} (raw model output: {m.group(0)})")


def take_screenshot(path: str = "", region: str = "") -> str:
    """Capture the screen and save it to a PNG. Windows/macOS only.

    region is optional "x1,y1,x2,y2" to grab part of the screen. Returns the
    path, which you then pass to find_ui_element or look_at_image.
    """
    from PIL import ImageGrab
    import tempfile, time as _t
    box = None
    if region:
        try:
            box = tuple(int(v) for v in region.replace(" ", "").split(","))
            if len(box) != 4:
                return "region must be 'x1,y1,x2,y2'"
        except ValueError:
            return "region must be 'x1,y1,x2,y2' with integer pixels"
    try:
        im = ImageGrab.grab(bbox=box)
    except Exception as e:
        return (f"screen capture failed: {e}\n"
                "ImageGrab needs Windows or macOS; on headless Linux there is "
                "no screen to grab.")
    if not path:
        path = os.path.join(tempfile.gettempdir(), f"shot-{int(_t.time())}.png")
    im.save(path)
    return f"saved {path} ({im.size[0]}x{im.size[1]})"


def list_vision_models() -> str:
    """List models whose config includes a vision projector."""
    try:
        ms = [m.id for m in _client().models.list().data]
    except Exception as e:
        return f"could not reach {API_BASE}: {e}"
    return "configured models:\n" + "\n".join(f"  {m}" for m in ms) + \
           f"\n\ndefault vision model: {VISION_MODEL}"


def build_server():
    # mcp 2.x: FastMCP was replaced by MCPServer (same decorator shape).
    from mcp.server import MCPServer
    srv = MCPServer("local-vision",
                    instructions="Look at screenshots using the local vision "
                                 "model, and locate UI elements to click.")
    srv.add_tool(take_screenshot)
    srv.add_tool(look_at_image)
    srv.add_tool(find_ui_element)
    srv.add_tool(list_vision_models)
    return srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", metavar="IMAGE")
    ap.add_argument("--target", default="Connect to the wireless network")
    a = ap.parse_args()
    if a.selftest:
        print("list_vision_models():");  print(list_vision_models()[:400]); print()
        print("look_at_image():");       print(look_at_image(a.selftest, "What application is this?")); print()
        print("find_ui_element():");     print(find_ui_element(a.selftest, a.target))
        sys.exit(0)
    build_server().run()
