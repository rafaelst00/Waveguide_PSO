from pathlib import Path
import time
import pyautogui
from pywinauto.application import Application
from pywinauto.keyboard import send_keys

def screenshot_abec_window(win):
    rect = win.rectangle()

    img = pyautogui.screenshot(
        region=(
            rect.left,
            rect.top,
            rect.width(),
            rect.height(),
        )
    )

    return img


def has_green_marker(img, x, y, box_size=12):
    """
    Checks if a small region around x/y contains a green marker.
    Coordinates are relative to the ABEC window screenshot.
    """

    crop = img.crop(
        (
            x - box_size,
            y - box_size,
            x + box_size,
            y + box_size,
        )
    ).convert("RGB")

    green_pixels = 0

    for r, g, b in crop.getdata():
        if g > 120 and r < 120 and b < 120:
            green_pixels += 1

    return green_pixels > 5


def get_abec_status_markers(win):
    img = screenshot_abec_window(win)

    width, height = img.size
    y = height - 18

    status = {
        "solved": has_green_marker(img, 40, y),
        "fields": has_green_marker(img, 220, y),
        "spectra_ok": has_green_marker(img, 400, y),
    }

    return status


def wait_until_solved_green(win, timeout=300, interval=1):
    start = time.time()

    while time.time() - start < timeout:
        status = get_abec_status_markers(win)

        if status["solved"]:
            print("Solved marker is green.")
            return True

        time.sleep(interval)

    raise TimeoutError("ABEC did not reach green Solved status.")


def wait_until_spectra_ok_green(win, timeout=120, interval=1):
    start = time.time()

    while time.time() - start < timeout:
        status = get_abec_status_markers(win)

        if status["spectra_ok"]:
            print("Spectra OK marker is green.")
            return True

        time.sleep(interval)

    raise TimeoutError("ABEC did not reach green Spectra OK status.")



def run_abec_simulation(
    abec_path: str | Path,
    project_path: str | Path,
    result_path: str | Path,
    solve_wait: float = 400,
    spectrum_wait: float = 100,
    buffer_wait: float = 2,
    short_wait: float = 1,
    remove_old_result: bool = True,
    close_after_export: bool = True,
) -> str:
    """
    Opens ABEC, loads a project, solves it, calculates spectrum,
    exports spectrum as text, and returns the exported text.
    """

    abec_path = Path(abec_path)
    project_path = Path(project_path)
    result_path = Path(result_path)

    if not abec_path.exists():
        raise FileNotFoundError(f"ABEC executable not found: {abec_path}")

    if not project_path.exists():
        raise FileNotFoundError(f"ABEC project not found: {project_path}")

    result_path.parent.mkdir(parents=True, exist_ok=True)

    if remove_old_result and result_path.exists():
        result_path.unlink()

    print("Starting ABEC...")
    app = Application(backend="uia").start(
        f'"{abec_path}" "{project_path}"'
    )

    time.sleep(buffer_wait)

    win = app.top_window()
    win.set_focus()
    print("ABEC window:", win.window_text())


    print("Starting solver...")
    win.set_focus()
    send_keys("{F5}")

    time.sleep(buffer_wait)

    print("Confirming solver warning...")
    send_keys("{ENTER}")
    
    wait_until_solved_green(win, timeout=solve_wait)
    time.sleep(short_wait)

    print("Calculating spectrum...")
    win.set_focus()
    send_keys("{F7}")

    wait_until_spectra_ok_green(win, timeout=spectrum_wait)
    time.sleep(short_wait)

    print("Exporting spectrum...")
    win.set_focus()
    send_keys("^{F7}")

    time.sleep(buffer_wait)

    print("Confirming export...")
    send_keys("{ENTER}")

    time.sleep(buffer_wait)

    if not result_path.exists():
        raise FileNotFoundError(f"Export failed. Result file not found: {result_path}")

    print("Export successful:", result_path)

    result_text = result_path.read_text(errors="ignore")

    if close_after_export:
        print("Closing ABEC...")

        try:
            win.close()
            time.sleep(short_wait)

            # confirm possible save/close dialog
            send_keys("{ENTER}")
            time.sleep(short_wait)

        except Exception:
            pass

        try:
            app.kill()
        except Exception:
            pass

    return result_text


