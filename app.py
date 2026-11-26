import os, glob, io, time, base64, random, json
import html as _html
import numpy as np, cv2, streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from PIL import Image, ImageOps
from matplotlib import cm
import baseline_patchcore as pc, module2 as m2, kb
import guard, shift_detector as sd

st.set_page_config(page_title="Two-Brain Inspector", layout="wide")

CSS = """
<style>
.stApp{background:#04070d!important;color:#e6edf7}
[data-testid="stAppViewContainer"],[data-testid="stHeader"]{background:transparent!important}
[data-testid="stSidebar"]{background:rgba(8,13,24,.93)!important;border-right:1px solid rgba(0,255,200,.15)}
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp p,.stApp label,.stApp li,
[data-testid="stMarkdownContainer"],[data-testid="stCaptionContainer"]{color:#e6edf7}
/* ---- readability: always light text on dark panels, sidebar above background ---- */
[data-testid="stSidebar"]{position:relative;z-index:20}
[data-testid="stHeader"]{z-index:30}
[data-testid="stSidebar"] *{color:#e6edf7}
[data-testid="stSidebar"] [data-baseweb="select"] *,[data-testid="stSidebar"] input{color:#e6edf7!important}
.block-container{background:rgba(4,7,13,.72);border-radius:14px;padding-top:2rem}
input,textarea,[data-baseweb="select"]>div,[data-baseweb="input"]{background:#111a2b!important;color:#e6edf7!important}
[data-baseweb="popover"] ul,[data-baseweb="menu"]{background:#111a2b!important}
[data-baseweb="popover"] li,[data-baseweb="menu"] li{color:#e6edf7!important}
.stButton button,.stDownloadButton button{background:#14233d!important;color:#e6edf7!important;
border:1px solid rgba(0,255,200,.45)!important}
.stButton button:hover,.stDownloadButton button:hover{border-color:#00ffc8!important;color:#fff!important}
button[data-baseweb="tab"] p{color:#cfe3ff!important;font-weight:600}
pre,code{background:#0d1526!important;color:#e6edf7!important}
[data-testid="stChatMessage"]{background:rgba(255,255,255,.05)}
/* ---- info / warning / error boxes: dark panel, light text ---- */
[data-testid="stAlert"],[data-testid="stAlertContainer"],div[data-baseweb="notification"]{
background:rgba(17,26,43,.95)!important;border:1px solid rgba(0,255,200,.45)!important;
border-radius:10px!important}
[data-testid="stAlert"] *,[data-testid="stAlertContainer"] *,
div[data-baseweb="notification"] *{color:#e6edf7!important;background:transparent!important}
/* ---- file uploader: dark box, light text ---- */
[data-testid="stFileUploader"] section,[data-testid="stFileUploaderDropzone"]{
background:#111a2b!important;border:1px dashed rgba(0,255,200,.5)!important;border-radius:10px}
[data-testid="stFileUploader"] *{color:#e6edf7!important}
[data-testid="stFileUploader"] small{color:#9fb3d1!important}
[data-testid="stFileUploader"] button{background:#14233d!important;
border:1px solid rgba(0,255,200,.45)!important}
[data-testid="stFileUploaderFile"],[data-testid="stFileUploaderFileData"]{background:transparent!important}
/* ---- toast popups ---- */
[data-testid="stToast"],[data-testid="stToast"] *{background:#111a2b!important;color:#e6edf7!important}
.hero{background:linear-gradient(90deg,#0b1f45,#1d3f7a,#0b1f45);
background-size:200% 100%;animation:grad 6s ease infinite,fadein .8s ease both;
padding:20px 26px;border-radius:14px;color:white;margin-bottom:14px;
border:1px solid rgba(0,255,200,.25);box-shadow:0 0 28px rgba(0,255,200,.12)}
.hero h1{margin:0;font-size:34px;color:white}
.hero p{margin:4px 0 0 0;opacity:.85;color:white}
@keyframes grad{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
.fade{animation:fadein .9s ease both}
@keyframes fadein{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
.scanbox{position:relative;overflow:hidden;border-radius:10px}
.scanbox img{width:100%;display:block;border-radius:10px}
.scanline{position:absolute;left:0;width:100%;height:4px;background:rgba(0,255,200,.95);
box-shadow:0 0 14px 6px rgba(0,255,200,.6);animation:scan 1.5s ease-in-out 2}
@keyframes scan{0%{top:0}50%{top:100%}100%{top:0}}
.reticle{position:absolute;width:46px;height:46px;border:3px solid #ff3b3b;border-radius:50%;
transform:translate(-50%,-50%);opacity:0;animation:ret 1.4s ease-out infinite;animation-delay:var(--d)}
.reticle::after{content:"";position:absolute;left:50%;top:50%;width:8px;height:8px;
margin:-4px 0 0 -4px;background:#ff3b3b;border-radius:50%}
@keyframes ret{0%{transform:translate(-50%,-50%) scale(2.2);opacity:0}25%{opacity:1}
100%{transform:translate(-50%,-50%) scale(.8);opacity:.9}}
.cap{text-align:center;font-size:13px;opacity:.7;margin-top:4px}
.badge{padding:14px;border-radius:12px;text-align:center;font-size:30px;font-weight:700;
color:white;margin-bottom:12px;animation:pop .5s ease both}
@keyframes pop{0%{transform:scale(.6);opacity:0}70%{transform:scale(1.08)}100%{transform:scale(1);opacity:1}}
.PASS{background:#2e9e5b}
.REVIEW{background:#e0a100}
.FAIL{background:#d43f3f;animation:pop .5s ease both,pulse 1.2s infinite .5s}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(212,63,63,.7)}70%{box-shadow:0 0 0 18px rgba(212,63,63,0)}
100%{box-shadow:0 0 0 0 rgba(212,63,63,0)}}
.lbl{font-size:14px;margin:8px 0 3px 0}
.bar{height:14px;background:rgba(128,128,128,.25);border-radius:8px;overflow:hidden}
.fill{height:100%;border-radius:8px;animation:grow 1.2s ease both}
@keyframes grow{from{width:0}}
.gauge{position:relative;width:200px;height:100px;overflow:hidden;margin:0 auto}
.gring{width:200px;height:200px;border-radius:50%;
-webkit-mask:radial-gradient(circle,transparent 56%,#000 57%);
mask:radial-gradient(circle,transparent 56%,#000 57%)}
.needle{position:absolute;left:calc(50% - 2px);bottom:0;width:4px;height:86px;background:#fff;
border-radius:3px;transform-origin:bottom center;transform:rotate(var(--a));
box-shadow:0 0 6px rgba(0,0,0,.6);animation:sweep 1.6s cubic-bezier(.2,1.3,.4,1) both}
@keyframes sweep{from{transform:rotate(-90deg)}to{transform:rotate(var(--a))}}
.hub{position:absolute;left:calc(50% - 8px);bottom:-8px;width:16px;height:16px;background:#fff;border-radius:50%}
.gnum{text-align:center;font-size:26px;font-weight:700;margin-top:4px}
.type{font-family:monospace;font-size:15px;overflow:hidden;white-space:nowrap;
border-right:2px solid #00ffc8;width:0;max-width:100%;
animation:typing 2.4s steps(40) .3s forwards,blink .8s step-end infinite}
@keyframes typing{from{width:0}to{width:calc(var(--n)*1ch)}}
@keyframes blink{50%{border-color:transparent}}
.zoom{border:2px solid #00ffc8;border-radius:8px;animation:glow 1.6s ease-in-out infinite alternate;
width:100%;display:block}
@keyframes glow{from{box-shadow:0 0 4px rgba(0,255,200,.3)}to{box-shadow:0 0 18px rgba(0,255,200,.9)}}
.hit{padding:10px;border-radius:10px;font-weight:700;color:white;text-align:center;animation:pop .5s ease both}
.hitY{background:#2e9e5b}.hitN{background:#d43f3f}.hitU{background:#6b7280}
.mini{display:inline-block;margin:3px;padding:4px 10px;border-radius:14px;font-size:12px;
color:white;animation:fadein .5s ease both}
.belt{display:flex;flex-wrap:wrap;gap:8px;padding:12px;border-radius:12px;
background:repeating-linear-gradient(90deg,rgba(128,128,128,.18) 0 20px,rgba(128,128,128,.08) 20px 40px);
background-size:40px 100%;animation:beltmove 1s linear infinite;min-height:130px}
@keyframes beltmove{from{background-position:0 0}to{background-position:40px 0}}
.chip{width:96px;border:3px solid;border-radius:8px;overflow:hidden;background:#111;animation:slidein .5s ease both}
.chip img{width:100%;display:block}
.chip span{display:block;text-align:center;font-size:11px;font-weight:700;color:white}
.chip.PASS{border-color:#2e9e5b}.chip.PASS span{background:#2e9e5b}
.chip.REVIEW{border-color:#e0a100}.chip.REVIEW span{background:#e0a100}
.chip.FAIL{border-color:#d43f3f}.chip.FAIL span{background:#d43f3f}
@keyframes slidein{from{transform:translateX(-70px);opacity:0}to{transform:none;opacity:1}}
.kpis{display:flex;gap:10px;flex-wrap:wrap;margin:8px 0}
.kpi{flex:1;min-width:110px;padding:10px;border-radius:10px;text-align:center;
background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.08);animation:fadein .5s ease both}
.kpi b{display:block;font-size:24px}
.kpi small{opacity:.7}
/* ---- alert banner ---- */
.alert{display:flex;align-items:center;gap:14px;padding:14px 18px;border-radius:12px;
margin:0 0 14px 0;color:white;font-weight:700;font-size:18px;animation:alin .5s ease both}
@keyframes alin{from{transform:translateY(-16px);opacity:0}to{transform:none;opacity:1}}
.alert small{display:block;font-weight:400;font-size:13px;opacity:.9}
.alert .lamp{width:22px;height:22px;border-radius:50%;background:#fff;flex:none}
.alert.PASS{background:linear-gradient(90deg,#1f7a46,#2e9e5b);box-shadow:0 0 22px rgba(46,158,91,.45)}
.alert.REVIEW{background:linear-gradient(90deg,#b07c00,#e0a100);box-shadow:0 0 22px rgba(224,161,0,.5)}
.alert.FAIL{background:linear-gradient(90deg,#a82020,#d43f3f);box-shadow:0 0 18px rgba(255,40,40,.45)}
.alog{display:flex;align-items:center;gap:10px;padding:6px 10px;margin:4px 0;border-radius:8px;
font-family:monospace;font-size:13px;background:rgba(255,255,255,.05);border-left:5px solid;
animation:fadein .4s ease both}
.alog.PASS{border-color:#2e9e5b}.alog.REVIEW{border-color:#e0a100}.alog.FAIL{border-color:#d43f3f}
/* ---- never fade stale elements while the script is running ---- */
.stApp [data-stale="true"],.stApp [data-stale],.stApp .stale-element,
[data-testid="stMarkdownContainer"],[data-testid="stElementContainer"]{opacity:1!important}
/* ---- defect report box: always light text on a dark panel ---- */
.report{white-space:pre-wrap;font-family:monospace;font-size:14px;color:#ffffff!important;
background:#0d1526!important;border:1px solid rgba(0,255,200,.4);border-radius:10px;padding:12px 14px}
.report *{color:#ffffff!important}
[data-testid="stText"],[data-testid="stText"] *,
[data-testid="stHeading"] *{color:#e6edf7!important}
</style>
"""
st.html(CSS)

# ---------------------------------------------------------------------
# Optional extra styling and effects from separate files (style.css, script.js)
# ---------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_css_file = os.path.join(_here, "style.css")
_js_file = os.path.join(_here, "script.js")
if os.path.exists(_css_file):
    with open(_css_file, encoding="utf-8") as _f:
        st.html(f"<style>{_f.read()}</style>")
if os.path.exists(_js_file):
    with open(_js_file, encoding="utf-8") as _f:
        components.html(f"<script>{_f.read()}</script>", height=0)

# ---------------------------------------------------------------------
# Light, animated look used ONLY on the AI assistant view
# ---------------------------------------------------------------------
LIGHT_CSS = """
<style>
@keyframes bgflow{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes bubblein{from{opacity:0;transform:translateY(16px) scale(.97)}to{opacity:1;transform:none}}
@keyframes aglow{0%,100%{box-shadow:0 0 0 rgba(0,255,200,0)}50%{box-shadow:0 0 18px rgba(0,255,200,.35)}}
@keyframes dotsb{0%,80%,100%{transform:scale(.4);opacity:.4}40%{transform:scale(1);opacity:1}}
@keyframes floaty{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}

/* keep the bottom chat bar see-through so the scene shows */
[data-testid="stBottom"],[data-testid="stBottom"]>div,[data-testid="stBottomBlockContainer"]{
background:transparent!important}

/* light text on dark background */
[data-testid="stMain"] h1,[data-testid="stMain"] h2,[data-testid="stMain"] h3,
[data-testid="stMain"] h4,[data-testid="stMain"] p,[data-testid="stMain"] li,
[data-testid="stMain"] label,[data-testid="stMain"] span,
[data-testid="stMain"] [data-testid="stMarkdownContainer"],
[data-testid="stMain"] [data-testid="stCaptionContainer"],
[data-testid="stMain"] [data-testid="stChatMessageContent"] *,
[data-testid="stMain"] [data-testid="stRadio"] *{color:#e6edf7!important}
.hero,.hero h1,.hero p{color:#ffffff!important}

/* buttons */
.stButton button{background:#14233d!important;color:#e6edf7!important;
border:1px solid rgba(0,255,200,.45)!important;border-radius:20px;transition:transform .15s,box-shadow .15s}
.stButton button p{color:#e6edf7!important}
.stButton button:hover{transform:translateY(-2px);box-shadow:0 6px 16px rgba(0,255,200,.25);
border-color:#00ffc8!important}

/* chat bubbles */
[data-testid="stChatMessage"]{background:rgba(255,255,255,.06)!important;
border:1px solid rgba(0,255,200,.25);border-radius:16px;animation:bubblein .5s ease-out both}
[data-testid="stChatMessage"]:hover{animation:aglow 1.8s ease-in-out infinite}
pre,code{background:#0d1526!important;color:#e6edf7!important}

/* chat input */
[data-testid="stChatInput"],[data-testid="stChatInput"]>div{background:#111a2b!important;
border-radius:16px}
[data-testid="stChatInput"]{border:1px solid rgba(0,255,200,.5);box-shadow:0 4px 18px rgba(0,255,200,.12)}
[data-testid="stChatInput"] textarea{background:transparent!important;color:#e6edf7!important;
-webkit-text-fill-color:#e6edf7!important}
[data-testid="stChatInput"] textarea::placeholder{color:#8fa3c4!important;opacity:1;
-webkit-text-fill-color:#8fa3c4!important}
[data-testid="stChatInput"] button{background:#14233d!important;color:#e6edf7!important}

/* thinking dots */
.dots{padding:6px 0}
.dots span{display:inline-block;width:10px;height:10px;margin:0 4px;border-radius:50%;
background:#00ffc8;animation:dotsb 1.2s infinite ease-in-out}
.dots span:nth-child(2){animation-delay:.2s}.dots span:nth-child(3){animation-delay:.4s}
.botbadge{display:inline-block;font-size:30px;animation:floaty 2.4s ease-in-out infinite}
</style>
"""

# ---------------------------------------------------------------------
# Dark animated background: "Digital twin inspection bay"
# ---------------------------------------------------------------------
BG_CSS = """
<style>
.block-container{position:relative;z-index:1}
.bgwrap{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden;
background:radial-gradient(ellipse at 50% 0%,#0b1630 0%,#04070d 70%)}
.bgwrap *{pointer-events:none;box-sizing:border-box}
.bgwrap .au{position:absolute;border-radius:50%;filter:blur(90px);opacity:.20}
.bgwrap .a1{width:520px;height:340px;background:#0e7490;left:-120px;top:8%;animation:d1 20s ease-in-out infinite alternate}
.bgwrap .a2{width:460px;height:320px;background:#6d28d9;right:-100px;top:40%;animation:d2 24s ease-in-out infinite alternate}
.bgwrap .a3{width:420px;height:280px;background:#047857;left:30%;bottom:10%;animation:d3 28s ease-in-out infinite alternate}
@keyframes d1{to{transform:translate(280px,160px) scale(1.2)}}
@keyframes d2{to{transform:translate(-300px,-140px) scale(1.2)}}
@keyframes d3{to{transform:translate(-200px,-120px) scale(.85)}}
.bgwrap .floor{position:absolute;left:0;right:0;bottom:70px;height:46%;perspective:420px;overflow:hidden;
-webkit-mask-image:linear-gradient(transparent,#000 55%);mask-image:linear-gradient(transparent,#000 55%)}
.bgwrap .plane{position:absolute;top:0;left:-50%;width:200%;height:200%;transform:rotateX(68deg);
transform-origin:top center;
background-image:linear-gradient(rgba(0,255,200,.22) 2px,transparent 2px),
linear-gradient(90deg,rgba(0,255,200,.22) 2px,transparent 2px);
background-size:70px 70px;animation:floorrun 1.8s linear infinite}
@keyframes floorrun{to{background-position:0 70px}}
.bgwrap .rain{position:absolute;top:-45%;font:12px monospace;writing-mode:vertical-rl;
color:rgba(0,255,200,.30);text-shadow:0 0 6px rgba(0,255,200,.5);animation:fall linear infinite}
.bgwrap .rain.r{color:rgba(255,90,90,.65);text-shadow:0 0 8px rgba(255,60,60,.8)}
@keyframes fall{to{transform:translateY(190vh)}}
.bgwrap .holo{position:absolute;right:6%;bottom:165px;width:120px;height:120px;perspective:700px}
.bgwrap .cube{position:relative;width:120px;height:120px;transform-style:preserve-3d;
animation:rot 14s linear infinite}
@keyframes rot{from{transform:rotateX(-22deg) rotateY(0)}to{transform:rotateX(-22deg) rotateY(360deg)}}
.bgwrap .f{position:absolute;width:120px;height:120px;border:1px solid rgba(0,255,200,.8);
background:rgba(0,255,200,.05);box-shadow:inset 0 0 18px rgba(0,255,200,.2)}
.bgwrap .f1{transform:translateZ(60px)}
.bgwrap .f2{transform:rotateY(180deg) translateZ(60px)}
.bgwrap .f3{transform:rotateY(90deg) translateZ(60px)}
.bgwrap .f4{transform:rotateY(-90deg) translateZ(60px)}
.bgwrap .f5{transform:rotateX(90deg) translateZ(60px)}
.bgwrap .f6{transform:rotateX(-90deg) translateZ(60px)}
.bgwrap .sp{position:absolute;width:120px;height:120px;background:rgba(0,255,200,.28);
border:1px solid #00ffc8;transform:rotateX(90deg) translateZ(-55px);animation:spm 2.6s ease-in-out infinite alternate}
@keyframes spm{to{transform:rotateX(90deg) translateZ(55px)}}
.bgwrap .core{position:absolute;left:50px;top:50px;width:20px;height:20px;border-radius:50%;
background:#ff3b3b;box-shadow:0 0 22px 6px rgba(255,60,60,.8);animation:corep 1.3s ease-in-out infinite}
@keyframes corep{0%,100%{opacity:.25;transform:scale(.7)}50%{opacity:1;transform:scale(1.2)}}
.bgwrap .hl{position:absolute;right:calc(6% - 20px);bottom:140px;width:160px;text-align:center;
font:11px monospace;color:rgba(0,255,200,.85);letter-spacing:2px}
.bgwrap .radar{position:absolute;right:-130px;top:-130px;width:440px;height:440px;border-radius:50%;
background:repeating-radial-gradient(circle,transparent 0 38px,rgba(0,255,200,.20) 39px 40px)}
.bgwrap .radar::before{content:"";position:absolute;inset:0;border-radius:50%;
background:conic-gradient(from 0deg,rgba(0,255,200,.38),transparent 70deg,transparent 360deg);
animation:spin 5s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.bgwrap .blip{position:absolute;width:9px;height:9px;border-radius:50%;background:#ff4b4b;
box-shadow:0 0 12px #ff4b4b;animation:bl 5s ease-in-out infinite}
@keyframes bl{0%,60%,100%{opacity:0}70%{opacity:1}}
.bgwrap .laser{position:absolute;top:0;bottom:0;left:0;width:2px;
background:linear-gradient(180deg,transparent,rgba(0,255,200,.8),transparent);
box-shadow:0 0 22px 7px rgba(0,255,200,.22);animation:ls 10s ease-in-out infinite alternate}
@keyframes ls{from{transform:translateX(0)}to{transform:translateX(100vw)}}
.bgwrap .hud{position:absolute;width:46px;height:46px;border:3px solid rgba(0,255,200,.6);animation:hp 2.4s ease-in-out infinite}
@keyframes hp{0%,100%{opacity:.35}50%{opacity:1}}
.bgwrap .h1{left:14px;top:14px;border-right:0;border-bottom:0}
.bgwrap .h2{right:14px;top:14px;border-left:0;border-bottom:0}
.bgwrap .h3{left:14px;bottom:14px;border-right:0;border-top:0}
.bgwrap .h4{right:14px;bottom:14px;border-left:0;border-top:0}
.bgwrap .belt2{position:absolute;left:0;bottom:0;width:100%;height:70px;border-top:3px solid rgba(0,255,200,.6);
background:repeating-linear-gradient(90deg,rgba(60,72,95,.75) 0 18px,rgba(22,27,40,.8) 18px 36px);
animation:br 1s linear infinite}
@keyframes br{to{background-position:36px 0}}
.bgwrap .cam{position:absolute;left:calc(50% - 22px);bottom:215px;width:44px;height:22px;border-radius:6px;
background:rgba(0,255,200,.6);box-shadow:0 0 16px rgba(0,255,200,.8)}
.bgwrap .cone{position:absolute;left:calc(50% - 60px);bottom:73px;width:120px;height:142px;
background:linear-gradient(0deg,rgba(0,255,200,.32),rgba(0,255,200,.03));
clip-path:polygon(42% 0,58% 0,100% 100%,0 100%);animation:cp 1.2s ease-in-out infinite alternate}
@keyframes cp{from{opacity:.5}to{opacity:1}}
.bgwrap .slab{position:absolute;left:calc(50% + 34px);bottom:196px;font:11px monospace;
color:rgba(0,255,200,.9);animation:b3 1s step-end infinite}
@keyframes b3{50%{opacity:.2}}
.bgwrap .part{position:absolute;left:0;bottom:78px;width:42px;height:42px;background:#5b6b82}
.bgwrap .part.ok{animation:travel 18s linear infinite,chkok 18s linear infinite}
.bgwrap .part.bad{animation:travel 18s linear infinite,chkbad 18s linear infinite}
.bgwrap .sq{border-radius:8px}
.bgwrap .nut{border-radius:50%}
.bgwrap .nut::before{content:"";position:absolute;inset:13px;border-radius:50%;background:rgba(4,7,13,.9)}
.bgwrap .hex{clip-path:polygon(25% 3%,75% 3%,100% 50%,75% 97%,25% 97%,0 50%)}
.bgwrap .part::after{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
color:#fff;font-weight:700;font-size:22px;opacity:0;animation:mark 18s linear infinite;animation-delay:inherit}
.bgwrap .ok::after{content:"\\2713"}
.bgwrap .bad::after{content:"\\2715"}
@keyframes travel{from{transform:translateX(-10vw)}to{transform:translateX(110vw)}}
@keyframes chkok{0%,44%{background:#5b6b82;box-shadow:none}50%{background:#00ffc8;box-shadow:0 0 24px #00ffc8}
58%,100%{background:#2e9e5b;box-shadow:0 0 10px rgba(46,158,91,.8)}}
@keyframes chkbad{0%,44%{background:#5b6b82;box-shadow:none}50%{background:#00ffc8;box-shadow:0 0 24px #00ffc8}
58%,100%{background:#d43f3f;box-shadow:0 0 18px rgba(212,63,63,.95)}}
@keyframes mark{0%,55%{opacity:0}60%,100%{opacity:1}}
@media (prefers-reduced-motion:reduce){.bgwrap *{animation:none!important}}
</style>
"""


def build_bg(level):
    r = random.Random(11)
    rain = ""
    for i in range(24):
        txt = "".join(r.choice("01PF01AI") for _ in range(r.randint(10, 20)))
        red = " r" if r.random() < 0.2 else ""
        rain += (f'<span class="rain{red}" style="left:{r.randint(1, 98)}%;'
                 f'animation-duration:{r.randint(9, 20)}s;'
                 f'animation-delay:-{r.randint(0, 20)}s">{txt}</span>')
    shapes = ["sq", "nut", "hex"]
    items = ""
    for k in range(7):
        kind = "bad" if k in (2, 5) else "ok"
        items += (f'<div class="part {shapes[k % 3]} {kind}" '
                  f'style="animation-delay:-{k * 18 / 7:.2f}s"></div>')
    faces = "".join(f'<div class="f f{i}"></div>' for i in range(1, 7))
    return (BG_CSS + f'<div class="bgwrap" style="opacity:{level}">'
            '<div class="au a1"></div><div class="au a2"></div><div class="au a3"></div>'
            '<div class="floor"><div class="plane"></div></div>'
            + rain +
            '<div class="radar"></div>'
            '<div class="blip" style="right:150px;top:70px"></div>'
            '<div class="blip" style="right:60px;top:150px;animation-delay:-2s"></div>'
            '<div class="blip" style="right:210px;top:170px;animation-delay:-3.5s"></div>'
            '<div class="laser"></div>'
            f'<div class="holo"><div class="cube">{faces}<div class="sp"></div>'
            '<div class="core"></div></div></div>'
            '<div class="hl">DIGITAL TWIN</div>'
            '<div class="cone"></div><div class="cam"></div><div class="slab">SCANNING...</div>'
            '<div class="belt2"></div>' + items +
            '<div class="hud h1"></div><div class="hud h2"></div>'
            '<div class="hud h3"></div><div class="hud h4"></div></div>')


COLORS = {"PASS": "#2e9e5b", "REVIEW": "#e0a100", "FAIL": "#d43f3f"}


@st.cache_resource
def get_ext():
    return pc.Extractor()


@st.cache_resource
def load(prod):
    import torch
    try:
        d = torch.load(f"banks/{prod}_m2.pt", weights_only=False,
                       map_location="cpu", mmap=True)
    except Exception:
        d = torch.load(f"banks/{prod}_m2.pt", weights_only=False, map_location="cpu")
    # .float() converts banks saved in half precision back to full precision
    d["bank"] = d["bank"].float().to(pc.DEVICE)
    d["lbanks"] = {k: v.float().to(pc.DEVICE) for k, v in d["lbanks"].items()}
    return d


def inspect(rgb, d):
    mp, ml = m2.both_maps(get_ext(), d["bank"], d["lbanks"], [rgb])
    ph, lg = mp[0] / d["tp"], ml[0] / d["tl"]
    fmap = np.maximum(ph, lg) if d["pose_ok"] else ph
    s = float(fmap.max())
    y, x = np.unravel_index(fmap.argmax(), fmap.shape)
    is_logical = d["pose_ok"] and lg.max() > 1.15 * ph.max()
    kind = "logical" if is_logical else "physical"
    hf = np.sort(d["hold_fused"])
    p = (1 + np.sum(hf >= s)) / (len(hf) + 1)
    return dict(fmap=fmap, s=s, y=int(y), x=int(x), kind=kind, p=p,
                ph=float(ph.max()), lg=float(lg.max()))


def b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def bar(label, value, color):
    w = int(min(value / 1.5, 1.0) * 100)
    return (f'<div class="lbl">{label}: <b>{value:.2f}</b></div>'
            f'<div class="bar"><div class="fill" style="width:{w}%;background:{color}"></div></div>')


def gauge(s, lo):
    ang = min(s / 1.5, 1.0) * 180
    a_lo = lo / 1.5 * 180
    grad = (f"conic-gradient(from -90deg,#2e9e5b 0deg,#2e9e5b {a_lo}deg,"
            f"#e0a100 {a_lo}deg,#e0a100 120deg,#d43f3f 120deg,#d43f3f 180deg,transparent 180deg)")
    return (f'<div class="gauge"><div class="gring" style="background:{grad}"></div>'
            f'<div class="needle" style="--a:{ang - 90}deg"></div><div class="hub"></div></div>'
            f'<div class="gnum">{s:.2f}</div><div class="cap">fused score (FAIL at 1.00)</div>')


def make_overlay(rgb, fmap, strength, show_circle, x, y):
    hm = (cm.jet(np.clip(fmap / 1.5, 0, 1))[..., :3] * 255).astype(np.uint8)
    out = ((1 - strength) * rgb + strength * hm).astype(np.uint8)
    out = np.ascontiguousarray(out)
    if show_circle:
        cv2.circle(out, (x, y), 18, (255, 255, 255), 2)
    return out


def zoom_crop(rgb, x, y, half=32):
    h, w = rgb.shape[:2]
    x0 = int(min(max(x - half, 0), max(w - 2 * half, 0)))
    y0 = int(min(max(y - half, 0), max(h - 2 * half, 0)))
    crop = rgb[y0:y0 + 2 * half, x0:x0 + 2 * half]
    return cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC)


def zone_name(x, y):
    r = min(int(y / pc.IMG * 3), 2)
    c = min(int(x / pc.IMG * 3), 2)
    rows = ["top", "middle", "bottom"]
    cols = ["left", "center", "right"]
    if r == 1 and c == 1:
        return "center"
    return f"{rows[r]}-{cols[c]}"


def severity(s):
    return "low" if s < 1.2 else ("medium" if s < 1.5 else "high")


def read_rgb(path, bright):
    img = Image.open(path).convert("RGB")
    rgb = np.array(img.resize((pc.IMG, pc.IMG), Image.BILINEAR))
    if abs(bright - 1.0) > 1e-6:
        rgb = np.clip(rgb.astype(np.float32) * bright, 0, 255).astype(np.uint8)
    return rgb



VIEWER_HTML = """
<style>
html,body{margin:0;background:transparent;font-family:Segoe UI,Arial,sans-serif;color:#e6edf7}
.wrap{display:flex;gap:18px;align-items:center;justify-content:center;height:410px;overflow:hidden}
.stage{width:360px;height:360px;perspective:1000px;cursor:grab;position:relative;touch-action:none}
.stage:active{cursor:grabbing}
.box{position:absolute;left:50px;top:40px;width:260px;height:260px;transform-style:preserve-3d;
transform:rotateX(-12deg) rotateY(25deg)}
.face{position:absolute;backface-visibility:hidden;overflow:hidden;
box-shadow:0 0 22px __COLOR__}
.front,.back{width:260px;height:260px;border:3px solid __COLOR__;border-radius:6px}
.front{transform:translateZ(14px)}
.back{transform:rotateY(180deg) translateZ(14px)}
.front img,.back img{width:100%;height:100%;display:block}
.side{width:28px;height:260px;background:linear-gradient(180deg,#1d2b46,#0d1526);
border:1px solid __COLOR__}
.right{left:116px;transform:rotateY(90deg) translateZ(130px)}
.left{left:116px;transform:rotateY(-90deg) translateZ(130px)}
.cap2{width:260px;height:28px;background:#14233d;border:1px solid __COLOR__}
.top{top:116px;transform:rotateX(90deg) translateZ(130px)}
.bottom{top:116px;transform:rotateX(-90deg) translateZ(130px)}
.ret{position:absolute;width:40px;height:40px;border:3px solid #ff3b3b;border-radius:50%;
transform:translate(-50%,-50%);display:__RET__;animation:pl 1.2s ease-out infinite}
@keyframes pl{0%{transform:translate(-50%,-50%) scale(1.8);opacity:0}30%{opacity:1}
100%{transform:translate(-50%,-50%) scale(.7);opacity:.9}}
.shadow{position:absolute;left:60px;bottom:6px;width:240px;height:34px;border-radius:50%;
background:radial-gradient(ellipse,rgba(0,255,200,.35),transparent 70%);filter:blur(6px)}
.floor{position:absolute;left:-50px;top:292px;width:460px;height:320px;
transform:rotateX(72deg);transform-origin:top center;
background-image:linear-gradient(rgba(0,255,200,.45) 2px,transparent 2px),
linear-gradient(90deg,rgba(0,255,200,.45) 2px,transparent 2px);
background-size:36px 36px;animation:run 1.6s linear infinite;
-webkit-mask-image:linear-gradient(#000 15%,transparent 90%);mask-image:linear-gradient(#000 15%,transparent 90%)}
@keyframes run{to{background-position:0 36px}}
.gridov{position:absolute;inset:0;pointer-events:none;
background-image:linear-gradient(rgba(255,255,255,.38) 1px,transparent 1px),
linear-gradient(90deg,rgba(255,255,255,.38) 1px,transparent 1px);
background-size:32.5px 32.5px}
.cg{position:absolute;left:-40px;top:-40px;width:340px;height:340px;pointer-events:none;
border:1px solid rgba(0,255,200,.65);background-color:rgba(0,255,200,.025);
background-image:linear-gradient(rgba(0,255,200,.30) 1px,transparent 1px),
linear-gradient(90deg,rgba(0,255,200,.30) 1px,transparent 1px);background-size:34px 34px}
.c1{transform:translateZ(170px)}
.c2{transform:rotateY(180deg) translateZ(170px)}
.c3{transform:rotateY(90deg) translateZ(170px)}
.c4{transform:rotateY(-90deg) translateZ(170px)}
.c5{transform:rotateX(90deg) translateZ(170px)}
.c6{transform:rotateX(-90deg) translateZ(170px)}
body.nogrid .gridov{display:none}
body.nocage .cage{display:none}
.panel{width:190px}
.panel h3{margin:0 0 6px 0;font-size:18px}
.badge{display:inline-block;padding:6px 16px;border-radius:10px;background:__COLOR__;
color:#fff;font-weight:700;font-size:20px;margin-bottom:12px}
.panel p{margin:4px 0;font-size:13px;opacity:.85}
.btns{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}
button{background:#14233d;color:#e6edf7;border:1px solid rgba(0,255,200,.5);border-radius:16px;
padding:6px 12px;cursor:pointer;font-size:12px}
button:hover{border-color:#00ffc8;color:#fff}
</style>
<div class="wrap">
 <div class="stage" id="stage">
  <div class="floor"></div>
  <div class="box" id="box">
   <div class="face front"><img src="data:image/png;base64,__FRONT__"><div class="gridov"></div>
     <div class="ret" style="left:__PX__%;top:__PY__%"></div></div>
   <div class="face back"><img src="data:image/png;base64,__BACK__"><div class="gridov"></div></div>
   <div class="face side right"><div class="gridov"></div></div><div class="face side left"><div class="gridov"></div></div>
   <div class="face cap2 top"><div class="gridov"></div></div><div class="face cap2 bottom"><div class="gridov"></div></div>
   <div class="cg cage c1"></div><div class="cg cage c2"></div><div class="cg cage c3"></div>
   <div class="cg cage c4"></div><div class="cg cage c5"></div><div class="cg cage c6"></div>
  </div>
  <div class="shadow"></div>
 </div>
 <div class="panel">
  <h3>3D part viewer</h3>
  <div class="badge">__VERDICT__</div>
  <p>Drag to turn the part in any direction, even upside down.</p>
  <p>Front: camera image.<br>Back: anomaly heatmap.</p>
  <div class="btns">
   <button onclick="goFront()">Front</button>
   <button onclick="goBack()">Heatmap</button>
   <button onclick="toggleSpin()" id="sp">Pause spin</button>
   <button onclick="toggleGrid()" id="gb">Grid: on</button>
   <button onclick="toggleCage()" id="cb">Cage: on</button>
   <button onclick="goTop()">Top</button>
   <button onclick="goSide()">Side</button>
  </div>
 </div>
</div>
<script>
let rx=-12, ry=25, spin=true, drag=false, lx=0, ly=0;
const box=document.getElementById('box'), stage=document.getElementById('stage');
function draw(){box.style.transform='rotateX('+rx+'deg) rotateY('+ry+'deg)';}
function tick(){ if(spin&&!drag){ry+=0.45;} draw(); requestAnimationFrame(tick);}
stage.addEventListener('pointerdown',e=>{drag=true;lx=e.clientX;ly=e.clientY;stage.setPointerCapture(e.pointerId);});
stage.addEventListener('pointermove',e=>{ if(!drag)return;
  ry+=(e.clientX-lx)*0.6; rx-=(e.clientY-ly)*0.6; lx=e.clientX; ly=e.clientY;});
stage.addEventListener('pointerup',()=>{drag=false;});
function goFront(){spin=false;document.getElementById('sp').textContent='Resume spin';
  ry=Math.round(ry/360)*360; rx=0;}
function goBack(){spin=false;document.getElementById('sp').textContent='Resume spin';
  ry=Math.round(ry/360)*360+180; rx=0;}
function toggleSpin(){spin=!spin;document.getElementById('sp').textContent=spin?'Pause spin':'Resume spin';}
function goTop(){spin=false;document.getElementById('sp').textContent='Resume spin';rx=-90;}
function goSide(){spin=false;document.getElementById('sp').textContent='Resume spin';
  ry=Math.round(ry/360)*360+90; rx=0;}
function toggleCage(){const off=document.body.classList.toggle('nocage');
  document.getElementById('cb').textContent=off?'Cage: off':'Cage: on';}
function toggleGrid(){const off=document.body.classList.toggle('nogrid');
  document.getElementById('gb').textContent=off?'Grid: off':'Grid: on';}
tick();
</script>
"""


def viewer3d(rgb, overlay, v, x, y):
    """Draggable 3D slab: camera image on the front, heatmap on the back."""
    html = (VIEWER_HTML
            .replace("__COLOR__", COLORS[v])
            .replace("__FRONT__", b64(rgb))
            .replace("__BACK__", b64(overlay))
            .replace("__PX__", f"{100.0 * x / pc.IMG:.1f}")
            .replace("__PY__", f"{100.0 * y / pc.IMG:.1f}")
            .replace("__RET__", "none" if v == "PASS" else "block")
            .replace("__VERDICT__", v))
    components.html(html, height=420)


# ---------- alerts ----------
ALERT_TEXT = {
    "PASS": ("PART APPROVED", "All checks clear. Send the part down the line."),
    "REVIEW": ("HUMAN REVIEW REQUIRED", "Borderline score or lighting shift. Route to a person."),
    "FAIL": ("ALERT: DEFECT DETECTED", "Stop the part. Reject it and check the defect zone."),
}


def alert_html(v, detail=""):
    t, sub = ALERT_TEXT[v]
    extra = f" {detail}" if detail else ""
    return (f'<div class="alert {v}"><div class="lamp"></div>'
            f'<div>{t}<small>{sub}{extra}</small></div></div>')


def make_wav(seq, rate=22050, nonce=0):
    """Build a loud, clear beep sequence as WAV bytes. seq = [(freq, start, dur), ...]
    nonce makes each file slightly different so the browser always replays it."""
    import wave
    total = max(st_ + du for _, st_, du in seq) + 0.1 + (nonce % 50) * 0.002
    sig = np.zeros(int(total * rate), dtype=np.float32)
    for f, st_, du in seq:
        n = int(du * rate)
        t = np.arange(n) / rate
        tone = np.sign(np.sin(2 * np.pi * f * t)) * 0.35 + np.sin(2 * np.pi * f * t) * 0.35
        env = np.minimum(1.0, np.minimum(t / 0.01, (du - t) / 0.02))
        i0 = int(st_ * rate)
        sig[i0:i0 + n] += (tone * env).astype(np.float32)
    pcm = (np.clip(sig, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


SEQS = {
    "PASS": [(660, 0.00, 0.15), (880, 0.18, 0.22)],
    "REVIEW": [(520, 0.00, 0.22), (520, 0.32, 0.22)],
    "FAIL": [(880, 0.0, 0.22), (620, 0.25, 0.22), (880, 0.5, 0.22),
             (620, 0.75, 0.22), (880, 1.0, 0.22), (620, 1.25, 0.3)],
}


def play_sound(v):
    """One player slot in the sidebar; replaced on every call, so players never stack."""
    n = st.session_state["snd"] = st.session_state.get("snd", 0) + 1
    data = make_wav(SEQS[v], nonce=n)
    with sound_slot.container():
        st.caption(f"Alert sound: {v}")
        try:
            st.audio(data, format="audio/wav", autoplay=True)
        except TypeError:
            src_ = base64.b64encode(data).decode()
            components.html(f'<!-- {n} --><audio controls autoplay style="width:100%" '
                            f'src="data:audio/wav;base64,{src_}"></audio>', height=50)


def log_alert(key, prod, v, s, zone):
    """Record one alert per new inspection and report whether it is new."""
    if st.session_state.get("last_key") == key:
        return False
    st.session_state["last_key"] = key
    al = st.session_state.setdefault("alerts", [])
    al.append(dict(t=time.strftime("%H:%M:%S"), prod=prod, v=v, s=s, zone=zone))
    st.session_state["alerts"] = al[-40:]
    return True


# ---------- AI assistant ----------
def local_answer(q, c, alerts):
    t = q.lower()
    if not c:
        return ("I don't have an inspection yet. Open the 'Inspect one part' view, "
                "choose an image, and then ask me about it.")
    v, s = c["v"], c["s"]
    if any(w in t for w in ["why", "explain", "reason", "how come"]):
        why = {"PASS": f"The fused score {s:.2f} is below REVIEW ({c['lo']:.2f}), so nothing looks unusual.",
               "REVIEW": f"The score {s:.2f} sits between REVIEW ({c['lo']:.2f}) and FAIL (1.00), "
                         "or the lighting shift check fired, so a person should look.",
               "FAIL": f"The score {s:.2f} is at or above 1.00, which means the part is further from "
                       "the good-part memory than almost any good part."}[v]
        return (f"{why} The physical brain scored {c['ph']:.2f} and the logical brain {c['lg']:.2f}; "
                f"the stronger one is the '{c['kind']}' brain. Note: {c['note'] or 'none'}.")
    if any(w in t for w in ["do next", "fix", "advice", "action", "what should", "repair", "recommend"]):
        return f"Recommended action for {c['prod']}:\n\n{c['advice']}"
    if any(w in t for w in ["where", "location", "zone", "position"]):
        if v == "PASS":
            return "No defect location, because the part passed."
        return (f"The hottest point is at x={c['x']}, y={c['y']}, in the {c['zone']} zone. "
                "The red reticle and the zoom card show it.")
    if any(w in t for w in ["p-value", "p value", "conformal", "probab"]):
        return (f"The conformal p-value is {c['p']:.3f}. About {100 * c['p']:.1f}% of known good parts "
                "score this high or higher. A small p-value means a good part would rarely look like this.")
    if any(w in t for w in ["light", "shift", "bright", "drift"]):
        if c["shift_ok"] is False:
            return "No shift profile exists yet. Run: python shift_detector.py fit"
        return (f"Shift check: {'FLAGGED' if c['shift_flag'] else 'OK'} "
                f"(max {c['zmax']:.1f} sigma, limit {c['zthr']:.1f}). When the lighting changes a lot, "
                "the app sends the part to REVIEW instead of trusting the score.")
    if any(w in t for w in ["review", "threshold", "false", "sensitiv", "alarm"]):
        return (f"REVIEW starts at {c['lo']:.2f} and FAIL at 1.00. Raising the REVIEW value lowers false "
                "alarms on good parts but lets borderline defects pass; lowering it catches more defects "
                "and sends more good parts to a human. Use the Conveyor scan view to see the trade-off.")
    if any(w in t for w in ["brain", "physical", "logical"]):
        return (f"The physical brain looks for local damage like scratches and dents (score {c['ph']:.2f}). "
                f"The logical brain checks that parts are in the right place and pose (score {c['lg']:.2f}). "
                "The fused score is the larger of the two.")
    if any(w in t for w in ["score", "gauge", "mean", "severity"]):
        return (f"The fused score is {s:.2f}. Under {c['lo']:.2f} is PASS, up to 1.00 is REVIEW, and 1.00 or more "
                f"is FAIL. Severity is {severity(s)} (low under 1.2, medium under 1.5, high above).")
    if any(w in t for w in ["summary", "session", "history", "alert", "log", "recent"]):
        if not alerts:
            return "No alerts logged yet."
        n = {k: sum(1 for a in alerts if a["v"] == k) for k in ("PASS", "REVIEW", "FAIL")}
        last = alerts[-1]
        return (f"This session: {n['PASS']} PASS, {n['REVIEW']} REVIEW, {n['FAIL']} FAIL. "
                f"The latest was {last['v']} on {last['prod']} at {last['t']} (score {last['s']:.2f}).")
    return (f"Current result: {c['prod']} is {v} with score {s:.2f}. You can ask me: why this verdict, "
            "what to do next, where the defect is, what the p-value means, whether lighting affected it, "
            "how to change REVIEW, or for a session summary.")


def llm_answer(q, c, alerts, chat, key):
    """Optional: use Claude if an API key is provided. Returns None on any problem."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        system = ("You are the assistant inside a factory defect inspector called Two-Brain Inspector. "
                  "It uses a physical (patch anomaly) brain and a logical (position) brain, fused score, "
                  "FAIL at 1.00, REVIEW from a configurable level, and a lighting shift guard. "
                  "Answer briefly and plainly for an operator. Only use the facts below; if something is "
                  "not in them, say you don't know.\n"
                  f"Current inspection: {json.dumps(c, default=str)}\n"
                  f"Recent alerts: {json.dumps(alerts[-8:], default=str)}")
        msgs = [{"role": m["role"], "content": m["content"]} for m in chat[-8:]]
        msgs.append({"role": "user", "content": q})
        resp = client.messages.create(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
            max_tokens=500, system=system, messages=msgs)
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    except Exception:
        return None


def typewriter(text):
    for w in text.split(" "):
        yield w + " "
        time.sleep(0.03)


DOTS = '<div class="dots"><span></span><span></span><span></span></div>'

# ---------- sidebar ----------
st.sidebar.header("Settings")
# Use the small demo dataset when it exists (for deployment); otherwise the full dataset.
_demo = os.path.join("demo_data", "mvtec_ad")
data_root = st.sidebar.text_input(
    "MVTec AD folder", _demo if os.path.isdir(_demo) else os.path.join("data", "mvtec_ad"))
prods = sorted(f.split("_m2")[0] for f in os.listdir("banks") if f.endswith("_m2.pt"))
if not prods:
    st.error("No fitted banks found in banks/.")
    st.stop()
prod = st.sidebar.selectbox("Product", prods)
d = load(prod)
test_dir = os.path.join(data_root, prod, "test")

src = st.sidebar.radio("Image source", ["Dataset test image", "Upload image"])
source_slot = st.sidebar.container()
pose_txt = f" (pose ratio {d['pose_ratio']:.3f})" if "pose_ratio" in d else ""
st.sidebar.markdown(f"Logical brain: **{'ON' if d['pose_ok'] else 'OFF'}**{pose_txt}")

rgb, label, mask_path, path = None, "", None, ""
st.sidebar.markdown("---")
st.sidebar.subheader("Safety checks")
bright = st.sidebar.slider("Simulate lighting (1.0 = normal)", 0.4, 1.6, 1.0, 0.05)
zthr = st.sidebar.slider("Shift sensitivity (z limit)", 2.0, 8.0, 4.0, 0.5)
lo_default = min(max(guard.load_lo(prod, 0.90), 0.50), 0.99)
lo = st.sidebar.slider("REVIEW starts at", 0.50, 0.99, float(lo_default), 0.01, key=f"lo_{prod}")
st.sidebar.subheader("Look and alerts")
strength = st.sidebar.slider("Heatmap strength", 0.1, 0.9, 0.45, 0.05)
animate = st.sidebar.checkbox("Scan animation", value=True)
show3d = st.sidebar.checkbox("3D part viewer", value=True)
show_gt = st.sidebar.checkbox("Show ground-truth mask", value=False)
bg_on = st.sidebar.checkbox("Background animation", value=True)
bg_level = st.sidebar.slider("Background intensity", 0.2, 1.0, 0.6, 0.05)
sound_on = st.sidebar.checkbox("Alert sound", value=True)
sound_slot = st.sidebar.empty()          # one slot for the alert player
test_col = st.sidebar.columns(3)
for _v, _c in zip(("PASS", "REVIEW", "FAIL"), test_col):
    if _c.button(f"🔊 {_v[0]}", key=f"tb_{_v}", help=f"Test {_v} sound"):
        play_sound(_v)
st.sidebar.subheader("AI assistant")
api_key = st.sidebar.text_input("Claude API key (optional)", type="password",
                                value=os.environ.get("ANTHROPIC_API_KEY", ""),
                                help="Without a key the assistant answers from built-in rules.")

# the dark animated background is not drawn on the AI assistant view
cur_view = st.session_state.get("view", "")
if bg_on:
    st.html(build_bg(bg_level))

# ---------- image source ----------
if src == "Dataset test image":
    if not os.path.isdir(test_dir):
        st.error(f"Not found: {test_dir}. Fix the MVTec AD folder.")
        st.stop()
    folders = sorted(os.listdir(test_dir))
    folder = source_slot.selectbox("Test folder", folders)
    files = sorted(glob.glob(os.path.join(test_dir, folder, "*.png")))
    if not files:
        st.error(f"No images in {os.path.join(test_dir, folder)}.")
        st.stop()
    idx = 0 if len(files) == 1 else source_slot.slider("Image index", 0, len(files) - 1, 0)
    path = files[idx]
    rgb = read_rgb(path, bright)
    label = f"dataset label: {folder}"
    mp = os.path.join(data_root, prod, "ground_truth", folder,
                      os.path.basename(path)[:-4] + "_mask.png")
    mask_path = mp if os.path.exists(mp) else None
else:
    up = source_slot.file_uploader("Upload image", type=["png", "jpg", "jpeg", "bmp", "webp"],
                                   key="uploader")
    if up is not None:
        try:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(up.getvalue()))).convert("RGB")
            rgb = np.array(img.resize((pc.IMG, pc.IMG), Image.BILINEAR))
            if abs(bright - 1.0) > 1e-6:
                rgb = np.clip(rgb.astype(np.float32) * bright, 0, 255).astype(np.uint8)
            label = "uploaded"
            path = f"upload_{up.name}_{up.size}"
        except Exception as e:
            rgb = None
            source_slot.error(f"Could not read this image: {e}")
if rgb is not None and abs(bright - 1.0) > 1e-6:
    label += f", brightness x{bright:.2f}"

# ---------- header ----------
st.markdown(
    '<div class="hero"><h1>Two-Brain Inspector</h1>'
    '<p>Trained on GOOD images only. Physical brain + position-aware '
    'logical brain + shift guard + AI assistant.</p></div>',
    unsafe_allow_html=True)

# radio instead of st.tabs: only the visible view runs on each rerun
view = st.radio("View", ["🔍 Inspect one part", "🏭 Conveyor scan", "🤖 AI assistant"],
                horizontal=True, label_visibility="collapsed", key="view")
in_assistant = view.startswith("🤖")

# =====================================================================
# VIEW 1: single image
# =====================================================================
if view.startswith("🔍"):
    if rgb is None:
        st.info("Choose a file in the sidebar, under Image source > Upload image, to start." if src == "Upload image" else "No image found.")
    else:
        alert_slot = st.empty()
        view3d_slot = st.empty()
        c1, c2, c3 = st.columns([1, 1, 1])
        slot = st.empty()
        akey = f"{prod}|{path}|{bright:.2f}"

        # animate only for a new image, not for every slider tweak
        if animate and st.session_state.get("anim_key") != akey:
            steps = ["Extracting features...", "Physical brain checking...",
                     "Logical brain checking...", "Checking lighting shift...", "Fusing scores..."]
            pb = slot.progress(0, text=steps[0])
            for i, t in enumerate(steps):
                pb.progress(int((i + 1) / len(steps) * 100), text=t)
                time.sleep(0.25)
        st.session_state["anim_key"] = akey

        # run the model only when the image changes
        if st.session_state.get("rk") != akey:
            st.session_state["rr"] = inspect(rgb, d)
            st.session_state["rk"] = akey
        r = st.session_state["rr"]
        sh = sd.check(rgb, prod, zthr)
        v, note = guard.decide(r["s"], lo, sh["flag"])
        slot.empty()
        zone = zone_name(r["x"], r["y"])

        # alert banner + sound + toast: once per new image or new verdict
        detail = "" if v == "PASS" else f"Zone: {zone}. Score {r['s']:.2f}."
        alert_slot.markdown(alert_html(v, detail), unsafe_allow_html=True)
        key = f"{prod}|{path}|{v}"
        new_alert = log_alert(key, prod, v, r["s"], zone)
        if new_alert and sound_on:
            play_sound(v)

        scan = '<div class="scanline"></div>' if animate else ""
        ret = ""
        if v != "PASS":
            px = 100.0 * r["x"] / pc.IMG
            py = 100.0 * r["y"] / pc.IMG
            delay = "3s" if animate else "0s"
            ret = (f'<div class="reticle" style="left:{px:.1f}%;top:{py:.1f}%;--d:{delay}"></div>')
        c1.markdown(
            f'<div class="scanbox fade"><img src="data:image/png;base64,{b64(rgb)}">'
            f'{scan}{ret}</div><div class="cap">Input ({label})</div>',
            unsafe_allow_html=True)

        out = make_overlay(rgb, r["fmap"], strength, v != "PASS", r["x"], r["y"])
        if show3d:
            with view3d_slot.container():
                viewer3d(rgb, out, v, r["x"], r["y"])
        c2.markdown(
            f'<div class="fade" style="animation-delay:.6s">'
            f'<img style="width:100%;border-radius:10px" src="data:image/png;base64,{b64(out)}">'
            f'<div class="cap">Anomaly heatmap</div></div>',
            unsafe_allow_html=True)

        with c3:
            st.markdown(f'<div class="badge {v}">{v}</div>', unsafe_allow_html=True)
            st.markdown(gauge(r["s"], lo), unsafe_allow_html=True)
            st.markdown(bar("Physical brain", r["ph"], "#3b82f6")
                        + bar("Logical brain", r["lg"], "#a855f7"), unsafe_allow_html=True)
            st.caption(f"FAIL at 1.00, REVIEW from {lo:.2f}")
            st.write(f"Conformal p-value: {r['p']:.3f} "
                     f"(about {100 * r['p']:.1f}% of good parts score this high or more)")
            if not sh["ok"]:
                st.caption("No shift profile yet. Run: python shift_detector.py fit")
            elif sh["flag"]:
                st.warning(f"Distribution shift: {sh['worst']} is {sh['zmax']:.1f} "
                           f"sigma away from training images.")
            else:
                st.caption(f"Shift check OK (max {sh['zmax']:.1f} sigma, limit {zthr:.1f})")

        if note:
            st.info(note)

        if animate and new_alert:
            if v == "PASS":
                st.toast("Part passed inspection", icon="✅")
            elif v == "FAIL":
                st.toast("Defect detected!", icon="🚨")
            else:
                st.toast("Needs human review", icon="⚠️")

        st.markdown("### Insight cards")
        k1, k2, k3 = st.columns([1, 1, 1])

        with k1:
            st.markdown("**Defect zoom**")
            if v == "PASS":
                st.caption("Nothing to zoom into: part is clean.")
            else:
                z = zoom_crop(rgb, r["x"], r["y"])
                st.markdown(f'<img class="zoom" src="data:image/png;base64,{b64(z)}">'
                            f'<div class="cap">4x zoom around the hottest point</div>',
                            unsafe_allow_html=True)

        with k2:
            st.markdown("**Live findings**")
            if v == "PASS":
                line = "PASS | no defect | part is clean"
            else:
                line = f"{v} | {r['kind']} | {severity(r['s'])} | zone: {zone}"
            st.markdown(f'<div class="type" style="--n:{len(line)}">{line}</div>',
                        unsafe_allow_html=True)
            st.markdown("**Localization check**")
            if mask_path and src == "Dataset test image":
                gt = Image.open(mask_path).convert("L").resize(
                    (r["fmap"].shape[1], r["fmap"].shape[0]), Image.NEAREST)
                gt = np.array(gt) > 0
                hit = bool(gt[r["y"], r["x"]])
                pred = r["fmap"] >= 1.0
                inter = np.logical_and(pred, gt).sum()
                union = np.logical_or(pred, gt).sum()
                iou = inter / union if union else 0.0
                cls = "hitY" if hit else "hitN"
                txt = "HIT: peak is inside the defect" if hit else "MISS: peak is outside the defect"
                st.markdown(f'<div class="hit {cls}">{txt}</div>', unsafe_allow_html=True)
                st.caption(f"Overlap (IoU) of FAIL zone with true defect: {iou:.2f}")
            else:
                st.markdown('<div class="hit hitU">No ground-truth mask for this image</div>',
                            unsafe_allow_html=True)

        with k3:
            st.markdown("**Session log**")
            hist = st.session_state.setdefault("hist", {})
            hist[key] = (prod, label, v, r["s"])
            chips = ""
            for (pr, lb, vv, sc) in list(hist.values())[-12:][::-1]:
                chips += (f'<span class="mini" style="background:{COLORS[vv]}">'
                          f'{pr} {vv} {sc:.2f}</span>')
            st.markdown(chips, unsafe_allow_html=True)
            if st.button("Clear log"):
                st.session_state["hist"] = {}
                st.session_state["alerts"] = []

        if show_gt and mask_path:
            gtimg = Image.open(mask_path).resize((pc.IMG, pc.IMG))
            st.image(gtimg, caption="Ground-truth mask", width=300)

        # alert center
        st.markdown("### Alert center")
        als = st.session_state.get("alerts", [])
        cnt = {k: sum(1 for a in als if a["v"] == k) for k in ("PASS", "REVIEW", "FAIL")}
        st.markdown(
            '<div class="kpis">'
            f'<div class="kpi"><b style="color:#2e9e5b">{cnt["PASS"]}</b><small>PASS</small></div>'
            f'<div class="kpi"><b style="color:#e0a100">{cnt["REVIEW"]}</b><small>REVIEW</small></div>'
            f'<div class="kpi"><b style="color:#d43f3f">{cnt["FAIL"]}</b><small>FAIL alerts</small></div>'
            '</div>', unsafe_allow_html=True)
        rows = ""
        for a in als[-6:][::-1]:
            rows += (f'<div class="alog {a["v"]}"><b>{a["t"]}</b><span>{a["v"]}</span>'
                     f'<span>{a["prod"]}</span><span>score {a["s"]:.2f}</span>'
                     f'<span>{a["zone"] if a["v"] != "PASS" else "-"}</span></div>')
        st.markdown(rows or "No alerts yet.", unsafe_allow_html=True)

        st.subheader("Defect report")
        if v == "PASS":
            st.write("No defect detected.")
            advice = "No action needed."
        else:
            st.write(f"**defect kind:** {r['kind']}  |  **severity:** {severity(r['s'])}")
            advice = kb.advise(prod, r["kind"], r["y"], r["x"], pc.IMG)
            # single-line HTML so Markdown cannot split the box; <br> for line breaks
            advice_html = _html.escape(str(advice)).replace("\n", "<br>")
            st.markdown(
                '<div class="report" style="font-family:monospace;font-size:15px;'
                'color:#ffffff;opacity:1;background:#0d1526;'
                'border:1px solid rgba(0,255,200,.4);border-radius:10px;'
                'padding:12px 14px;line-height:1.7">'
                f'{advice_html}</div>',
                unsafe_allow_html=True)

        # context for the AI assistant view
        st.session_state["ctx"] = dict(
            prod=prod, v=v, s=round(r["s"], 3), ph=round(r["ph"], 3), lg=round(r["lg"], 3),
            kind=r["kind"], x=r["x"], y=r["y"], zone=zone, p=round(float(r["p"]), 4),
            lo=lo, note=note or "", advice=advice, shift_ok=bool(sh["ok"]),
            shift_flag=bool(sh["flag"]), zmax=float(sh.get("zmax", 0.0)), zthr=zthr,
            label=label)

        report = (f"Two-Brain Inspector report\n"
                  f"Product: {prod}\nImage: {label}\nVerdict: {v}\n"
                  f"Fused score: {r['s']:.3f}\nPhysical: {r['ph']:.3f}\n"
                  f"Logical: {r['lg']:.3f}\nKind: {r['kind']}\n"
                  f"Severity: {severity(r['s'])}\n"
                  f"Location: x={r['x']}, y={r['y']} ({zone})\n"
                  f"Conformal p-value: {r['p']:.4f}\n"
                  f"Shift check: {'FLAGGED' if sh['flag'] else 'OK'}\n"
                  f"Note: {note or '-'}\nAdvice: {advice}\n")
        dl1, dl2 = st.columns(2)
        dl1.download_button("⬇ Download report (.txt)", report, file_name=f"{prod}_report.txt")
        dl2.download_button("⬇ Download heatmap (.png)", png_bytes(out),
                            file_name=f"{prod}_heatmap.png", mime="image/png")

# =====================================================================
# VIEW 2: conveyor scan
# =====================================================================
if view.startswith("🏭"):
    st.write("Random parts from the test set roll past the scanner. "
             "Each one is judged with your current REVIEW and shift settings.")
    if src != "Dataset test image":
        st.info("Switch Image source to 'Dataset test image' to use the conveyor scan.")
    else:
        b1, b2 = st.columns(2)
        n = b1.slider("Parts on the belt", 6, 60, 16)
        seed = b2.number_input("Random seed", 0, 999, 7)
        go = st.button("▶ Start conveyor scan")
        if go:
            allf = []
            for fo in sorted(os.listdir(test_dir)):
                for f in sorted(glob.glob(os.path.join(test_dir, fo, "*.png"))):
                    allf.append((fo, f))
            rnd = random.Random(int(seed))
            rnd.shuffle(allf)
            batch = allf[:n]

            conv_alert = st.empty()
            kpi_slot = st.empty()
            belt_slot = st.empty()
            prog = st.progress(0, text="Starting belt...")
            res = []
            t0 = time.perf_counter()

            def kpis(res):
                tot = len(res)
                np_ = sum(1 for x in res if x["v"] == "PASS")
                nr = sum(1 for x in res if x["v"] == "REVIEW")
                nf = sum(1 for x in res if x["v"] == "FAIL")
                defects = [x for x in res if x["fo"] != "good"]
                caught = sum(1 for x in defects if x["v"] != "PASS")
                goods = [x for x in res if x["fo"] == "good"]
                false_al = sum(1 for x in goods if x["v"] != "PASS")
                pct = f"{100 * caught / len(defects):.0f}%" if defects else "-"
                items = [("Scanned", tot), ("PASS", np_), ("REVIEW", nr), ("FAIL", nf),
                         ("Defects caught", f"{caught}/{len(defects)} ({pct})"),
                         ("Good flagged", f"{false_al}/{len(goods)}")]
                html = '<div class="kpis">'
                for k, val in items:
                    html += f'<div class="kpi"><b>{val}</b><small>{k}</small></div>'
                return html + "</div>"

            def belt(res):
                html = '<div class="belt">'
                for x in res[::-1][:24]:
                    html += (f'<div class="chip {x["v"]}">'
                             f'<img src="data:image/png;base64,{x["th"]}">'
                             f'<span>{x["v"]}</span></div>')
                return html + "</div>"

            fail_sounded = False
            for i, (fo, f) in enumerate(batch):
                prog.progress(int(i / len(batch) * 100),
                              text=f"Scanning part {i + 1} of {len(batch)} ({fo})")
                im = read_rgb(f, bright)
                rr = inspect(im, d)
                shh = sd.check(im, prod, zthr)
                vv, _ = guard.decide(rr["s"], lo, shh["flag"])
                ov = make_overlay(im, rr["fmap"], strength, vv != "PASS", rr["x"], rr["y"])
                th = cv2.resize(ov, (96, 96), interpolation=cv2.INTER_AREA)
                res.append(dict(fo=fo, v=vv, s=rr["s"], th=b64(th)))
                if vv == "FAIL":
                    conv_alert.markdown(alert_html("FAIL", f"Part {i + 1} ({fo}) rejected."),
                                        unsafe_allow_html=True)
                    # beep as soon as the first reject appears
                    if sound_on and not fail_sounded:
                        play_sound("FAIL")
                        fail_sounded = True
                kpi_slot.markdown(kpis(res), unsafe_allow_html=True)
                belt_slot.markdown(belt(res), unsafe_allow_html=True)

            prog.progress(100, text="Belt finished")
            secs = time.perf_counter() - t0
            st.caption(f"{len(res)} parts in {secs:.1f}s ({1000 * secs / len(res):.0f} ms per part)")

            df = pd.DataFrame({"score": [x["s"] for x in res]})
            df["FAIL line"] = 1.0
            df["REVIEW line"] = lo
            st.line_chart(df)

            defects = [x for x in res if x["fo"] != "good"]
            missed = sum(1 for x in defects if x["v"] == "PASS")
            nfail = sum(1 for x in res if x["v"] == "FAIL")
            nrev = sum(1 for x in res if x["v"] == "REVIEW")
            if missed:
                conv_alert.markdown(alert_html("FAIL", f"{missed} defect(s) slipped through as PASS."),
                                    unsafe_allow_html=True)
                st.toast(f"{missed} defect(s) slipped through as PASS", icon="⚠️")
                if sound_on:
                    play_sound("FAIL")
            elif nfail or nrev:
                conv_alert.markdown(alert_html("REVIEW", f"{nfail} rejected, {nrev} to review."),
                                    unsafe_allow_html=True)
                if defects:
                    st.toast("Every defect on the belt was caught", icon="🎯")
                if sound_on and not fail_sounded:
                    play_sound("REVIEW")
            else:
                conv_alert.markdown(alert_html("PASS", "Whole batch is clean."), unsafe_allow_html=True)
                if sound_on:
                    play_sound("PASS")
            st.session_state["batch"] = dict(n=len(res), fail=nfail, review=nrev, missed=missed)

# =====================================================================
# VIEW 3: AI assistant (light, animated look; no alerts or sounds here)
# =====================================================================
if in_assistant:
    sound_slot.empty()                       # remove any old alert player
    st.html(LIGHT_CSS)

    mode = "Claude (API key found)" if api_key else "Built-in rules (offline)"
    st.markdown(f'<span class="botbadge">🤖</span> **Ask about the part you just inspected.**  '
                f'Mode: `{mode}`', unsafe_allow_html=True)
    st.caption("Answers use your most recent inspection from the 'Inspect one part' view.")
    chat = st.session_state.setdefault("chat", [])
    box = st.container()
    quick = [("Why this verdict?", "Why this verdict?"),
             ("What should I do next?", "What should I do next?"),
             ("Where is the defect?", "Where is the defect?"),
             ("Session summary", "Give me a session summary")]
    cols = st.columns(len(quick))
    asked = None
    for col, (lab, q) in zip(cols, quick):
        if col.button(lab, key=f"q_{lab}"):
            asked = q
    typed = st.chat_input("Ask the inspector assistant...")
    asked = typed or asked

    with box:
        if not chat and not asked:
            st.caption("Try: 'Why did it fail?', 'What does the p-value mean?', "
                       "'Did lighting affect this?', 'How do I reduce false alarms?'")
        for m in chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if asked:
            ctx = st.session_state.get("ctx")
            alerts = st.session_state.get("alerts", [])
            with st.chat_message("user"):
                st.markdown(asked)
            with st.chat_message("assistant"):
                ph = st.empty()
                ph.markdown(DOTS, unsafe_allow_html=True)      # thinking dots
                ans = None
                if api_key and ctx:
                    ans = llm_answer(asked, ctx, alerts, chat, api_key)
                if ans is None:
                    ans = local_answer(asked, ctx, alerts)
                ph.empty()
                try:
                    st.write_stream(typewriter(ans))             # typed out word by word
                except AttributeError:
                    st.markdown(ans)
            chat.append({"role": "user", "content": asked})
            chat.append({"role": "assistant", "content": ans})
            st.session_state["chat"] = chat

    if chat and st.button("Clear chat"):
        st.session_state["chat"] = []
        st.rerun()
        