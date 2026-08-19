#!/usr/bin/env python3
"""MCP server exposing the local vision model as callable tools.

Lets a text agent (opencode, Claude Code, anything MCP-capable) look at
the live screen and interact with UI elements mid-reasoning.

Tools:
  look_at_screen(question)           screenshot + ask about it
  find_ui_element(target)           screenshot + GUI grounding -> pixel coords
  interact_ui_element(target, action, text)  screenshot + find + interact

Run directly for a smoke test:
  python mcp-vision.py --selftest
"""
import argparse, ast, base64, io, os, re, sys, tempfile, time as _t

API_BASE = os.environ.get("LLAMASWAP_URL", "http://mojs-ai.local:8080/v1")
VISION_MODEL = os.environ.get("VISION_MODEL", "os-atlas-base-7b-q6-k")
MAX_TOKENS = 900          # reasoning models return EMPTY below ~600


def _client():
    from openai import OpenAI
    return OpenAI(base_url=API_BASE, api_key="local")


def _grab_screen(region=None):
    """Take a screenshot, save to temp, return (path, original_size)."""
    from PIL import ImageGrab
    box = None
    if region:
        try:
            box = tuple(int(v) for v in region.replace(" ", "").split(","))
            if len(box) != 4:
                raise ValueError
        except ValueError:
            raise ValueError("region must be 'x1,y1,x2,y2' with integer pixels")
    try:
        im = ImageGrab.grab(bbox=box)
    except Exception as e:
        raise RuntimeError(
            f"screen capture failed: {e}\n"
            "ImageGrab needs Windows or macOS; on headless Linux there is "
            "no screen to grab."
        )
    path = os.path.join(tempfile.gettempdir(), f"shot-{int(_t.time())}.png")
    im.save(path)
    return path, im.size


def _save_image(im):
    """Save a PIL Image to temp, return (path, size)."""
    path = os.path.join(tempfile.gettempdir(), f"shot-{int(_t.time() * 1000)}.png")
    im.save(path)
    return path, im.size


def _scale_image(im, max_side):
    """Scale image so the longest side is max_side, preserving aspect ratio."""
    from PIL import Image
    w, h = im.size
    if max(w, h) <= max_side:
        return im
    scale = max_side / max(w, h)
    return im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _crop_image(im, cx, cy, radius_x, radius_y):
    """Crop a region around (cx, cy) with separate x/y radii in pixels."""
    from PIL import Image
    w, h = im.size
    x1 = max(0, int(cx - radius_x))
    y1 = max(0, int(cy - radius_y))
    x2 = min(w, int(cx + radius_x))
    y2 = min(h, int(cy + radius_y))
    return im.crop((x1, y1, x2, y2)), (x1, y1)


def _encode(path):
    from PIL import Image
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such image: {path}")
    im = Image.open(path)
    orig = im.size
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


def _parse_box(txt):
    """Extract bounding box from OS-Atlas output, return (x1, y1, x2, y2) in 0-1000 range.

    OS-Atlas returns coordinates in 0-1000 range. Formats:
      (x1,y1),(x2,y2)   — standard
      [[x1,y1,x2,y2]]   — bracket notation
      [x1,y1,x2,y2]     — single brackets
    """
    # Try bracket format first: [[x1,y1,x2,y2]] or [x1,y1,x2,y2]
    m = re.search(r"\[?\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]\]?", txt)
    if m:
        x1, y1, x2, y2 = [float(m.group(i)) for i in range(1, 5)]
        x1, y1, x2, y2 = [max(0.0, min(1000.0, v)) for v in [x1, y1, x2, y2]]
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    # Try standard format: (x1,y1),(x2,y2)
    p1 = r"\((\d+\.?\d*),(\d+\.?\d*)\)"
    m1 = re.search(p1, txt)
    if not m1:
        return None
    x1, y1 = float(m1.group(1)), float(m1.group(2))
    rest = txt[m1.end():]
    m2 = re.search(r",\((\d+\.?\d*),(\d+\.?\d*)\)", rest)
    if not m2:
        return None
    x2, y2 = float(m2.group(1)), float(m2.group(2))
    x1, y1, x2, y2 = [max(0.0, min(1000.0, v)) for v in [x1, y1, x2, y2]]
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _coords_to_pixels(box_01, img_w, img_h, offset=(0, 0)):
    """Convert a 0-1000 box to pixel coords, with optional offset for cropped images."""
    x1, y1, x2, y2 = box_01
    ox, oy = offset
    px1 = x1 / 1000 * img_w + ox
    py1 = y1 / 1000 * img_h + oy
    px2 = x2 / 1000 * img_w + ox
    py2 = y2 / 1000 * img_h + oy
    return (min(px1, px2), min(py1, py2), max(px1, px2), max(py1, py2))


def _locate_element(target, screen_path=None, screen_im=None, screen_size=None):
    """Locate a UI element using the vision model.

    Returns (cx, cy, detail_str) or (None, None, error_str).
    """
    from PIL import Image

    if screen_im is None:
        screen_im = Image.open(screen_path)
    orig_w, orig_h = screen_im.size

    prompt = (f'This is a screenshot of a computer screen. '
              f'What is the position of the element corresponding to the command '
              f'"{target}" (with bbox)?')

    # Step 1: rough localization at 1280px
    scaled = _scale_image(screen_im, 1280)
    scaled_path, _ = _save_image(scaled)
    txt, _, _ = _ask_vision(scaled_path, prompt)
    box = _parse_box(txt)
    if box is None:
        return None, None, f"No bounding box in model reply: {txt[:200]}"

    rough_cx = (box[0] + box[2]) / 2 / 1000 * orig_w
    rough_cy = (box[1] + box[3]) / 2 / 1000 * orig_h
    print(f"DEBUG step1: box={box} -> center=({rough_cx:.0f},{rough_cy:.0f})",
          file=sys.stderr)

    # Step 2: crop 15% around rough location, refine at 1280px
    crop_rx = int(orig_w * 0.15)
    crop_ry = int(orig_h * 0.15)
    crop_im, (crop_ox, crop_oy) = _crop_image(screen_im, rough_cx, rough_cy, crop_rx, crop_ry)
    crop_scaled = _scale_image(crop_im, 1280)
    crop_path, _ = _save_image(crop_scaled)
    txt2, _, _ = _ask_vision(crop_path, prompt)
    box2 = _parse_box(txt2)

    if box2 is None:
        cx, cy = rough_cx, rough_cy
        detail = (f"click_x={cx:.0f} click_y={cy:.0f} "
                  f"image={orig_w}x{orig_h} (refine failed) "
                  f"(raw: {txt[:120]})")
        return cx, cy, detail

    crop_w, crop_h = crop_im.size
    cx = (box2[0] + box2[2]) / 2 / 1000 * crop_w + crop_ox
    cy = (box2[1] + box2[3]) / 2 / 1000 * crop_h + crop_oy
    print(f"DEBUG step2: box={box2} -> center=({cx:.0f},{cy:.0f})",
          file=sys.stderr)

    detail = (f"click_x={cx:.0f} click_y={cy:.0f} "
              f"image={orig_w}x{orig_h} "
              f"(raw: {txt2[:120]})")
    return cx, cy, detail


# -- MCP tools -----------------------------------------------------------

def look_at_screen(question: str = "Describe what you see on screen.") -> str:
    """Take a screenshot and ask a question about what's visible.

    Use for: understanding screen content, reading text, checking state.
    NOT for: finding clickable elements (use find_ui_element or interact_ui_element).
    Works best with yes/no or factual questions. May give unreliable answers
    for open-ended descriptions like 'describe everything you see'.
    """
    try:
        path, (w, h) = _grab_screen()
    except Exception as e:
        return f"Screenshot failed: {e}"
    from PIL import Image
    im = Image.open(path)
    im = _scale_image(im, 1280)
    path, _ = _save_image(im)
    prompt = f"This is a screenshot of a computer screen. {question}"
    txt, size, orig = _ask_vision(path, prompt)
    return f"{txt}\n\n(screen {w}x{h}, sent as {size[0]}x{size[1]})"


def find_ui_element(target: str) -> str:
    """Take a screenshot and locate a UI element, returning pixel coordinates.

    Use for: finding where something is on screen before clicking.
    Returns detail string with click_x, click_y, box, and image size.
    target should be visually descriptive: 'the Save button in the toolbar',
    NOT just 'the button'.
    """
    try:
        path, (w, h) = _grab_screen()
    except Exception as e:
        return f"Screenshot failed: {e}"
    cx, cy, detail = _locate_element(target, screen_path=path)
    if cx is None:
        return f"Could not locate element: {detail}"
    return detail


def interact_ui_element(target: str, action: str = "click",
                        text: str = "") -> str:
    """Take a screenshot, find a UI element, and interact with it.

    Actions:
        click        - single click on the element center
        double_click - double-click on the element center
        right_click  - right-click on the element center (context menu)
        type         - click the element, then type the text
        hover        - move the mouse to the element center
        scroll       - scroll down at the element center
        key          - press a keyboard key (use text for key name, e.g. "enter")
    """
    import pyautogui
    pyautogui.FAILSAFE = True

    try:
        path, (w, h) = _grab_screen()
    except Exception as e:
        return f"Screenshot failed: {e}"

    cx, cy, detail = _locate_element(target, screen_path=path)
    if cx is None:
        return f"Could not locate element: {detail}"

    action = action.lower().strip()
    try:
        if action == "click":
            pyautogui.click(cx, cy)
            return f"Clicked ({cx:.0f}, {cy:.0f}). {detail}"

        elif action == "double_click":
            pyautogui.doubleClick(cx, cy)
            return f"Double-clicked ({cx:.0f}, {cy:.0f}). {detail}"

        elif action == "right_click":
            pyautogui.rightClick(cx, cy)
            return f"Right-clicked ({cx:.0f}, {cy:.0f}). {detail}"

        elif action == "type":
            pyautogui.click(cx, cy)
            _t.sleep(0.15)
            import pyperclip
            pyperclip.copy(text)
            pyautogui.hotkey("ctrl", "v")
            return f"Typed '{text}' at ({cx:.0f}, {cy:.0f}). {detail}"

        elif action == "hover":
            pyautogui.moveTo(cx, cy)
            return f"Hovered at ({cx:.0f}, {cy:.0f}). {detail}"

        elif action == "scroll":
            pyautogui.click(cx, cy)
            pyautogui.scroll(-3)
            return f"Scrolled down at ({cx:.0f}, {cy:.0f}). {detail}"

        elif action == "key":
            if not text:
                return "action=key requires the 'text' parameter with the key name"
            pyautogui.press(text.lower().strip())
            return f"Pressed key '{text}'. {detail}"

        else:
            return (f"Unknown action '{action}'. "
                    "Valid: click, double_click, right_click, type, hover, scroll, key")

    except Exception as e:
        return f"Interaction failed at ({cx:.0f}, {cy:.0f}): {e}\n{detail}"


# -- Server --------------------------------------------------------------

def build_server():
    from mcp.server import MCPServer
    srv = MCPServer("vision",
                    instructions=("Look at the live screen and interact with "
                                  "UI elements using the local vision model.\n\n"
                                  "Tools:\n"
                                  "  look_at_screen   — ask a question about what's on screen\n"
                                  "  find_ui_element  — locate a UI element, returns pixel coords\n"
                                  "  interact_ui_element — find + click/type/scroll/etc\n\n"
                                  "How to use effectively:\n"
                                  "- look_at_screen: best for yes/no or factual questions.\n"
                                  "  May give unreliable answers for open-ended descriptions.\n"
                                  "- find/interact_ui_element: target must describe VISUAL APPEARANCE.\n"
                                  "  GOOD: 'the red Submit button at the bottom of the form'\n"
                                  "  BAD:  'the button'\n"
                                  "- For title bar buttons, mention position (top-right, next to X).\n"
                                  "- All tools auto-screenshot — no need to screenshot separately.\n\n"
                                  "Common workflows:\n"
                                  "- Read text: look_at_screen('What is the title of...')\n"
                                  "- Click element: interact_ui_element('the Save button', 'click')\n"
                                  "- Find then click: find_ui_element first, then interact\n"
                                  "- Type text: interact_ui_element('the search box', 'type', 'query')\n"
                                  "- Scroll: interact_ui_element('the main content area', 'scroll')"))
    srv.add_tool(look_at_screen)
    srv.add_tool(find_ui_element)
    srv.add_tool(interact_ui_element)
    return srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        print("look_at_screen():")
        print(look_at_screen("What application is visible?")[:500])
        print()
        print("find_ui_element():")
        print(find_ui_element("the Start button or taskbar"))
        sys.exit(0)
    build_server().run()
