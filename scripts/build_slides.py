#!/usr/bin/env python3
"""Build slides/awesome_diffusion_OT_deck.html (single file, figures embedded) from slides/deck_content.json."""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
content = json.loads((ROOT / "slides/deck_content.json").read_text(encoding="utf-8"))


def img(path, style="max-height:62vh"):
    data = base64.b64encode((ROOT / path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" style="{style}">'


CSS = """
*{box-sizing:border-box}html,body{margin:0;height:100%;font-family:-apple-system,'PingFang SC','Helvetica Neue',Arial,sans-serif;background:#0f1419;color:#e6e9ef}
.deck{height:100vh;width:100vw;overflow:hidden;position:relative}
.slide{position:absolute;inset:0;padding:44px 64px;display:none;flex-direction:column}
.slide.active{display:flex}
h1{font-size:40px;margin:0 0 6px;color:#fff;font-weight:700;letter-spacing:.2px}
h2{font-size:30px;margin:0 0 18px;color:#fff;font-weight:650;border-left:6px solid #5aa9ff;padding-left:14px}
.sub{color:#9fb3c8;font-size:17px;margin-bottom:14px}
ul{font-size:20px;line-height:1.5;margin:6px 0 0 6px;padding-left:22px}li{margin:6px 0}li li{font-size:18px;color:#c9d3df}
.cols{display:flex;gap:28px;flex:1;min-height:0}.col{flex:1;min-width:0}
table{border-collapse:collapse;font-size:16px;width:100%;margin-top:8px}th,td{border:1px solid #2b3a4a;padding:6px 9px;text-align:left;vertical-align:top}th{background:#1b2735;color:#fff}
.fig{flex:1;display:flex;align-items:center;justify-content:center;min-height:0}.fig img{max-width:100%;object-fit:contain;border-radius:6px;background:#fff;padding:6px}
.note{position:absolute;bottom:18px;left:64px;right:64px;font-size:13px;color:#7d8fa3;display:flex;justify-content:space-between}
.tag{display:inline-block;background:#1f4e79;color:#fff;border-radius:4px;padding:1px 8px;font-size:14px;margin-right:6px}
.big{font-size:26px;line-height:1.55;color:#e6e9ef}.hl{color:#ffd166}.ok{color:#7bd88f}.bad{color:#ff7b7b}
.title{justify-content:center;align-items:flex-start}.title h1{font-size:54px;line-height:1.15}.title .sub{font-size:22px;margin-top:12px}
kbd{background:#1b2735;border:1px solid #2b3a4a;border-radius:4px;padding:0 6px}
"""
JS = """
let i=0;const s=document.querySelectorAll('.slide');function show(n){i=(n+s.length)%s.length;s.forEach((e,k)=>e.classList.toggle('active',k===i));document.querySelector('#pg').textContent=(i+1)+' / '+s.length;location.hash=i+1;}
document.addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key))show(i+1);if(['ArrowLeft','PageUp'].includes(e.key))show(i-1);if(e.key==='Home')show(0);if(e.key==='End')show(s.length-1);});
document.addEventListener('click',e=>{if(e.clientX>window.innerWidth*0.6)show(i+1);else if(e.clientX<window.innerWidth*0.15)show(i-1);});
show(parseInt(location.hash.slice(1)||'1')-1);
"""
slides = []
for k, sl in enumerate(content["slides"]):
    body = sl["body"]
    for key, path in sl.get("figs", {}).items():
        body = body.replace("{{" + key + "}}", img(path))
    cls = "slide title" if sl.get("kind") == "title" else "slide"
    head = f"<h1>{sl['title']}</h1>" if sl.get("kind") == "title" else f"<h2>{sl['title']}</h2>"
    sub = f"<div class='sub'>{sl['sub']}</div>" if sl.get("sub") else ""
    slides.append(f"<section class='{cls}'>{head}{sub}{body}<div class='note'><span>{content['footer']}</span><span id='pg'></span></div></section>")
html = f"<!doctype html><html lang='zh'><head><meta charset='utf-8'><title>{content['title']}</title><style>{CSS}</style></head><body><div class='deck'>{''.join(slides)}</div><script>{JS}</script></body></html>"
out = ROOT / "slides/awesome_diffusion_OT_deck.html"
out.write_text(html, encoding="utf-8")
print("wrote", out, f"{out.stat().st_size//1024} KB, {len(slides)} slides")
