import os, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path("/tmp/tsr/_review_out")
OUT = HERE / "stills"; OUT.mkdir(exist_ok=True)
ENTITIES = [("NHC_EP932026", "EP93"), ("NHC_AL902026", "AL90")]
PANELS = ["p-tracks", "p-board", "p-intensity", "p-ships"]

def shoot():
    with sync_playwright() as p:
        b = p.chromium.launch()
        for sid, tag in ENTITIES:
            url = (HERE / f"review_{sid}.html").as_uri()
            # desktop
            pg = b.new_page(viewport={"width": 1200, "height": 1000}, device_scale_factor=2)
            pg.goto(url, wait_until="networkidle", timeout=45000)
            pg.wait_for_function("document.querySelector('#tracks').innerHTML.length > 200", timeout=20000)
            time.sleep(1.2)  # font settle
            pg.screenshot(path=str(OUT / f"{tag}_desktop_full.png"), full_page=True)
            for pan in PANELS:
                el = pg.query_selector(f"#{pan}")
                if el: el.screenshot(path=str(OUT / f"{tag}_desktop_{pan}.png"))
            pg.close()
            # mobile
            mp = b.new_page(viewport={"width": 390, "height": 840}, device_scale_factor=2)
            mp.goto(url, wait_until="networkidle", timeout=45000)
            mp.wait_for_function("document.querySelector('#tracks').innerHTML.length > 200", timeout=20000)
            time.sleep(1.2)
            mp.screenshot(path=str(OUT / f"{tag}_mobile_full.png"), full_page=True)
            mp.close()
            print(f"shot {tag}")
        b.close()
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.name}  {f.stat().st_size//1024} KB")

if __name__ == "__main__":
    shoot()
