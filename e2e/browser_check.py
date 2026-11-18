"""Drive the running app in a real browser and check the main user journeys.

Needs pip install playwright reportlab and playwright install chromium.
Start the app first, for example::

    cd frontend && npm run build && cd ..
    python e2e/serve_stub.py --dist frontend/dist          # in one terminal
    python e2e/browser_check.py                            # in another

Prints one line per check and exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import expect, sync_playwright
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8765/")
parser.add_argument("--screenshots", type=Path, help="Folder to save desktop and mobile screenshots")
args = parser.parse_args()


def make_pdf(path: Path, pages: list[list[str]]) -> None:
    pdf = canvas.Canvas(str(path), pagesize=A4)
    for lines in pages:
        y = 800
        for line in lines:
            pdf.drawString(40, y, line)
            y -= 18
        pdf.showPage()
    pdf.save()


def make_fixtures(folder: Path) -> Path:
    make_pdf(folder / "handbook.pdf", [
        ["Photosynthesis is the process by which plants convert sunlight, water and carbon dioxide",
         "into glucose and oxygen. It takes place in the chloroplasts of leaf cells."],
        ["The French Revolution began in 1789 and ended the monarchy.",
         "The storming of the Bastille became its defining symbol."],
        ["A transistor is a semiconductor device used to amplify or switch electronic signals.",
         "Billions of transistors fit on a modern processor chip."],
    ])
    make_pdf(folder / "blank.pdf", [[], []])
    (folder / "notes.txt").write_text("not a pdf")
    return folder


results = []
def check(name, fn):
    try:
        fn(); results.append((name, "ok", ""))
    except Exception as exc:
        results.append((name, "FAIL", str(exc).split("\n")[0][:200]))

with sync_playwright() as p, tempfile.TemporaryDirectory() as tmp:
    FIX = make_fixtures(Path(tmp))
    def shot(name, full=False):
        if args.screenshots:
            args.screenshots.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(args.screenshots / f"{name}.png"), full_page=full)

    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    console_errors = []
    page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" and not m.text.startswith("Failed to load resource") else None)
    page.on("pageerror", lambda e: console_errors.append(str(e)))
    page.goto(args.url)

    check("empty state prompts to upload", lambda: expect(page.get_by_text("Upload a PDF on the left")).to_be_visible())
    check("question box disabled before upload", lambda: expect(page.get_by_label("Your question")).to_be_disabled())

    # upload non-PDF: rejected client-side with a message
    page.set_input_files("input[type=file]", str(FIX / "notes.txt"))
    check("non-PDF rejected with message", lambda: expect(page.locator(".status")).to_contain_text("only PDF files"))
    # upload scanned-style PDF: server error shown
    page.set_input_files("input[type=file]", str(FIX / "blank.pdf"))
    check("PDF without text shows server message", lambda: expect(page.locator(".status")).to_contain_text("no extractable text"))
    check("nothing listed after failures", lambda: expect(page.get_by_text("No documents yet.")).to_be_visible())

    # good upload
    page.set_input_files("input[type=file]", str(FIX / "handbook.pdf"))
    check("document appears in list", lambda: expect(page.locator(".doc-name")).to_have_text("handbook.pdf"))
    check("doc meta shows pages and passages", lambda: expect(page.locator(".doc-meta")).to_have_text("3 pages · 3 passages"))
    check("success status", lambda: expect(page.locator(".status")).to_contain_text("Document added."))
    check("question box enabled", lambda: expect(page.get_by_label("Your question")).to_be_enabled())
    check("send disabled while empty", lambda: expect(page.get_by_role("button", name="Ask", exact=True)).to_be_disabled())

    # ask a question with Enter
    box = page.get_by_label("Your question")
    box.fill("How do plants make glucose from sunlight?")
    box.press("Enter")
    check("user message shown", lambda: expect(page.locator(".message-user")).to_contain_text("How do plants make glucose"))
    check("assistant answer shown", lambda: expect(page.locator(".message-assistant .answer-text")).to_contain_text("Photosynthesis is the process"))
    check("citation chip rendered", lambda: expect(page.locator(".message-assistant a.cite")).to_have_count(1))
    check("source lists file and page", lambda: expect(page.locator(".sources summary")).to_contain_text("handbook.pdf"))
    check("source page number is 1", lambda: expect(page.locator(".sources .source-page")).to_have_text("page 1"))
    check("passage hidden until opened", lambda: expect(page.locator(".sources blockquote")).to_be_hidden())
    page.locator("a.cite").click()
    check("clicking citation opens the passage", lambda: expect(page.locator(".sources blockquote")).to_contain_text("chloroplasts"))
    check("input cleared after sending", lambda: expect(box).to_have_value(""))
    shot("desktop")

    # second question, other page
    box.fill("When did the French Revolution begin?")
    page.get_by_role("button", name="Ask", exact=True).click()
    check("second answer cites page 2", lambda: expect(page.locator(".message-assistant").nth(1).locator(".source-page")).to_have_text("page 2"))

    # not in document
    box.fill("quantum chromodynamics lagrangian")
    box.press("Enter")
    check("unanswerable question shows not-found note", lambda: expect(page.locator(".message-assistant").nth(2)).to_contain_text("could not find the answer"))
    check("no sources for unanswerable", lambda: expect(page.locator(".message-assistant").nth(2).locator(".sources")).to_have_count(0))

    # shift+enter makes a newline, not a send
    box.fill("line one"); box.press("Shift+Enter"); box.type("line two")
    check("shift+enter inserts newline", lambda: expect(box).to_have_value("line one\nline two"))
    box.fill("")

    # reload keeps documents (server persisted)
    page.reload()
    check("documents survive reload", lambda: expect(page.locator(".doc-name")).to_have_text("handbook.pdf"))

    # remove
    page.get_by_role("button", name="Remove handbook.pdf").click()
    check("remove empties list", lambda: expect(page.get_by_text("No documents yet.")).to_be_visible())
    check("question box disabled again", lambda: expect(page.get_by_label("Your question")).to_be_disabled())

    # mobile layout
    page.set_viewport_size({"width": 390, "height": 844})
    page.set_input_files("input[type=file]", str(FIX / "handbook.pdf"))
    expect(page.locator(".doc-name")).to_have_text("handbook.pdf")
    page.get_by_label("Your question").fill("What is a transistor?")
    page.get_by_label("Your question").press("Enter")
    expect(page.locator(".message-assistant")).to_be_visible()
    check("no horizontal scroll on mobile", lambda: (_ for _ in ()).throw(AssertionError("overflow")) if page.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth") else None)
    shot("mobile", full=True)

    # server error path: stop reaching the API
    page.route("**/api/ask", lambda route: route.abort())
    page.get_by_label("Your question").fill("anything")
    page.get_by_label("Your question").press("Enter")
    check("network failure shown as error message", lambda: expect(page.locator(".message-error")).to_contain_text("Could not reach the server"))

    check("no console errors", lambda: (_ for _ in ()).throw(AssertionError(console_errors)) if console_errors else None)
    browser.close()

width = max(len(n) for n, _, _ in results)
for n, s, d in results:
    print(f"{s:4}  {n:<{width}}  {d}")
print(f"\n{sum(1 for _,s,_ in results if s=='ok')}/{len(results)} passed")
