"""Visit the Streamlit app so Community Cloud doesn't put it to sleep.

Run by .github/workflows/keep_alive.yml every 3 hours.
If the app is already asleep, clicks the "Yes, get this app back up!" button.
"""
import sys

from playwright.sync_api import sync_playwright

URL = "https://realestateadvisor-vga53jpujrkqvfsxy2xt9d.streamlit.app/"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=120_000)
    page.wait_for_timeout(10_000)

    wake = page.get_by_role("button", name="Yes, get this app back up!")
    if wake.count() > 0:
        print("App was asleep - waking it up")
        wake.first.click()
        page.wait_for_timeout(90_000)          # give the app time to boot
    else:
        print("App is awake")

    # The app itself runs inside an iframe; wait until its title renders
    app = page.frame_locator("iframe[title='streamlitApp']")
    try:
        app.get_by_text("Investment Advisor").first.wait_for(timeout=120_000)
        print("App loaded OK")
    except Exception as exc:                    # report but don't hide failures
        print(f"App did not finish loading: {exc}")
        browser.close()
        sys.exit(1)
    browser.close()
