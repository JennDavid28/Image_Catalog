"""Browser smoke test for the UI (needs: pip install playwright && playwright install chromium).

    python scripts/ui_smoke.py              # run every check
    python scripts/ui_smoke.py --shots docs # also refresh the README screenshots

Starts its own server on a temp catalog, seeds the sample images and drives the real page.
"""
import json, os, socket, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
shots = Path(sys.argv[sys.argv.index("--shots") + 1]) if "--shots" in sys.argv else None
s = socket.socket(); s.bind(("127.0.0.1", 0)); PORT = s.getsockname()[1]; s.close()
BASE = f"http://127.0.0.1:{PORT}"
TOKEN = ""
def api(p):
    req = urllib.request.Request(BASE + "/api" + p)
    if TOKEN: req.add_header("Authorization", "Bearer " + TOKEN)
    return json.load(urllib.request.urlopen(req))
results = []


def check(name, cond, extra=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name, extra)


def shot(pg, name):
    if shots:
        shots.mkdir(exist_ok=True); pg.screenshot(path=str(shots / name))


tmp = tempfile.mkdtemp()
srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.api:app", "--port", str(PORT)], cwd=ROOT,
                       env={**os.environ, "CATALOG_DATA": tmp}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(60):
        try: api("/health"); break
        except Exception: time.sleep(.5)
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1500, "height": 900}, color_scheme="dark"); errs = []; media_errors = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("response", lambda r: media_errors.append(f"{r.status} {r.url}") if r.status >= 400 and any(x in r.url for x in ("/thumb", "/file", "/asset")) else None)
        pg.goto(BASE); pg.wait_for_selector("#dlg-auth[open]")
        pg.click("#tab-register"); pg.fill("#auth-username", "smokeuser"); pg.fill("#auth-dispname", "Smoke User")
        pg.fill("#auth-password", "smokepass123"); pg.click("#btn-auth-submit")
        time.sleep(1)
        if pg.locator("#dlg-auth").evaluate("e=>e.open"):
            print("Authentication form error:", pg.locator("#auth-err").inner_text(), "Page errors:", errs)
        pg.wait_for_function("!document.querySelector('#dlg-auth').open")
        TOKEN = pg.evaluate("localStorage.getItem('imgcatalog_token')")
        pg.wait_for_selector("#btn-seed"); pg.click("#btn-seed")
        pg.wait_for_selector("#film button", timeout=180000); time.sleep(1.5)
        pg.click("#user-profile-widget"); pg.wait_for_selector("#dlg-user-profile[open]")
        check("profile shows signed-in user", pg.inner_text("#pop-name") == "Smoke User")
        pg.click("#dlg-user-profile [data-close]")
        n0 = len(api("/images")); check("sample catalog loaded", n0 == 10, n0)
        pg.click("#film button >> nth=6"); time.sleep(1.5)                       # apple_orange
        pg.wait_for_function("document.querySelector('#img').naturalWidth > 0", timeout=15000)
        check("authenticated image and thumbnail requests load", not media_errors, media_errors)
        check("detector model and pipeline are identified separately",
              pg.get_by_text("Model used", exact=True).count() == 1 and pg.get_by_text("Pipeline", exact=True).count() == 1)
        shot(pg, "ui-catalog.png")

        pg.click(".cls[data-l='apple'] [data-pick]"); pg.wait_for_selector("#picker"); shot(pg, "ui-colour-picker.png")
        pg.click("#picker [data-c='#ec4899']"); pg.click("#pk-apply"); time.sleep(.6)
        check("class colour saved", next(c for c in api("/classes") if c["label"] == "apple")["color"] == "#ec4899")
        check("canvas redrew in the new colour", pg.locator("#svg path[fill='#ec4899']").count() == 1)
        b4 = pg.locator("#svg .mask").count(); pg.click(".cls[data-l='apple'] [data-eye]")
        check("eye hides the class", pg.locator("#svg .mask").count() == b4 - 1); pg.click(".cls[data-l='apple'] [data-eye]")
        z0 = pg.inner_text("#z-pct"); bx = pg.locator("#viewport").bounding_box(); pg.mouse.move(bx["x"] + 300, bx["y"] + 250); pg.mouse.wheel(0, -500); time.sleep(.3)
        check("wheel zoom", pg.inner_text("#z-pct") != z0); pg.click("#z-fit")

        pg.click("#t-crop"); pg.wait_for_selector("[data-cr]"); pg.click("[data-a='1']"); time.sleep(.3); shot(pg, "ui-crop.png")
        c = [int(pg.input_value(f"[data-cr='{i}']")) for i in range(4)]; check("crop keeps 1:1", abs(c[2] - c[3]) <= 2, c)
        pg.click("#apply-crop"); time.sleep(3); im = api("/images")
        check("crop saved as a copy, original kept", len(im) == n0 + 1 and im[-1]["user"]["display_name"].endswith("(cropped)"))
        w0 = im[-1]["auto"]["width"]; pg.click("#t-resize"); pg.wait_for_selector("#rs-w"); pg.click("[data-pct='50']"); pg.click("#apply-resize"); time.sleep(3)
        check("resize 50%", abs(api("/images")[-1]["auto"]["width"] - w0 / 2) <= 1)
        pg.click("#t-adjust"); pg.wait_for_selector("[data-adj='hue']")
        pg.eval_on_selector("[data-adj='hue']", "e=>{e.value=90;e.dispatchEvent(new Event('input',{bubbles:true}))}")
        check("live recolour preview", "hue-rotate(90deg)" in pg.eval_on_selector("#img", "e=>e.style.filter"))
        pg.click("#apply-adjust"); time.sleep(3); check("recolour saved", api("/images")[-1]["user"]["display_name"].endswith("(recoloured)"))

        pg.click("#film button >> nth=6"); time.sleep(1.5); pg.click(".obj:has-text('apple')"); pg.click("#ob-recolor"); pg.wait_for_selector("#rc-hex")
        pg.click("[data-rc='#3b82f6']"); pg.click("#apply-recolor"); time.sleep(3); check("object recolour", api("/images")[-1]["user"]["display_name"].endswith("(recoloured)"))
        pg.click("#film button >> nth=6"); time.sleep(1.5); pg.click(".obj:has-text('orange')"); pg.click("#ob-cut"); time.sleep(3)
        last = api("/images")[-1]; check("cut-out is a transparent PNG", last["user"]["display_name"].endswith("(cut-out)") and last["auto"]["format"] == "png")
        pg.click("#film button >> nth=6"); time.sleep(1.5); pg.click(".obj:has-text('apple')"); pg.click("#ob-erase"); time.sleep(3)
        check("erase object saved", api("/images")[-1]["user"]["display_name"].endswith("(object erased)"))

        n = len(api("/images")); pg.click("#t-trash"); time.sleep(1.5)
        check("delete goes to Trash", len(api("/images")) == n - 1 and len(api("/trash")["images"]) == 1)
        pg.click("#tab-trash"); pg.wait_for_selector(".tcard"); time.sleep(.5); shot(pg, "ui-trash.png")
        pg.click("[data-restore]"); time.sleep(1); check("restore", len(api("/images")) == n and not api("/trash")["images"])
        pg.click("#tab-catalog"); time.sleep(1); pg.click("#t-trash"); time.sleep(1.5); pg.click("#tab-trash"); pg.wait_for_selector(".tcard")
        pg.click("#tr-empty"); pg.click("#cf-ok"); time.sleep(1.2); check("empty trash", not api("/trash")["images"])

        pg.click("#tab-catalog"); time.sleep(1); pg.click("#film button >> nth=4"); time.sleep(1.5); pg.click(".obj >> nth=0"); pg.click("#ob-delete"); time.sleep(1)
        check("removed object is in Trash", len(api("/trash")["objects"]) == 1)
        t0 = pg.evaluate("document.documentElement.dataset.theme"); pg.click("#btn-theme")
        check("theme toggles", pg.evaluate("document.documentElement.dataset.theme") != t0)
        pg.click("#cap-pill"); pg.wait_for_selector("#caps-body"); check("capability dialog", "pip install" in pg.inner_text("#caps-body") or "active" in pg.inner_text("#caps-body"))
        pg.click("#dlg-caps [data-close]")
        pg.click("#user-profile-widget"); pg.wait_for_selector("#dlg-user-profile[open]"); pg.click("#btn-logout")
        pg.wait_for_function("!document.querySelector('#dlg-user-profile').open")
        pg.wait_for_selector("#dlg-auth[open]")
        check("logout returns to login", not pg.evaluate("localStorage.getItem('imgcatalog_token')"))
        check("no JavaScript errors", not errs, errs); b.close()
finally:
    srv.terminate()
print(f"{sum(results)}/{len(results)} passed"); sys.exit(0 if all(results) else 1)
