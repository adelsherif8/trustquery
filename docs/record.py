"""Record a demo video of TrustQuery using Playwright."""
import asyncio
from playwright.async_api import async_playwright

URL = "http://localhost:8085/"
OUT = "/Users/adel/Desktop/GHAI/trustquery/docs/video"


async def ask(page, q, wait_for=".headline, .err"):
    box = page.locator("#q")
    await box.click()
    await box.fill("")
    await box.type(q, delay=38)
    await page.wait_for_timeout(400)
    await page.click("#askbtn")
    try:
        await page.wait_for_selector(wait_for, timeout=20000)
    except Exception:
        pass
    await page.wait_for_timeout(2600)


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(
            viewport={"width": 1440, "height": 1000},
            record_video_dir=OUT,
            record_video_size={"width": 1440, "height": 1000},
            device_scale_factor=2,
        )
        page = await ctx.new_page()
        await page.goto(URL)
        await page.wait_for_timeout(1500)

        # 1) simple governed query
        await ask(page, "Total revenue by region")
        # 2) hard paraphrase -> shows the LLM planner
        await ask(page, "how full are our hotels in dubai across the months?")
        await page.evaluate("window.scrollTo({top:0,behavior:'smooth'})")
        await page.wait_for_timeout(800)
        # 3) guardrail: illegal grain rejected with explanation
        await ask(page, "show occupancy broken down by room type")
        await page.wait_for_timeout(1200)
        # 4) run the reconciliation eval suite
        await page.evaluate("document.querySelector('.evalbtn').scrollIntoView({block:'center'})")
        await page.wait_for_timeout(600)
        await page.click(".evalbtn")
        try:
            await page.wait_for_selector(".score", timeout=25000)
        except Exception:
            pass
        await page.wait_for_timeout(3200)

        await ctx.close()
        await browser.close()
        print("recorded")


asyncio.run(main())
