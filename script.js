/* Two-Brain Inspector - UI enhancements.
   Runs inside a components.html iframe and works on the parent Streamlit page. */
(function () {
  var doc;
  try {
    doc = window.parent.document;
  } catch (e) {
    return; // cross-origin: do nothing
  }
  if (doc.__tbInit) return; // run once
  doc.__tbInit = true;

  /* 1. Ripple effect on buttons */
  doc.addEventListener("click", function (e) {
    var btn = e.target.closest(".stButton > button, .stDownloadButton > button");
    if (!btn) return;
    var rect = btn.getBoundingClientRect();
    var size = Math.max(rect.width, rect.height);
    var r = doc.createElement("span");
    r.className = "tb-ripple";
    r.style.width = r.style.height = size + "px";
    r.style.left = e.clientX - rect.left - size / 2 + "px";
    r.style.top = e.clientY - rect.top - size / 2 + "px";
    btn.appendChild(r);
    setTimeout(function () { r.remove(); }, 650);
  });

  /* 2. Count-up animation for numeric metrics */
  function countUp(el) {
    if (el.dataset.tbDone) return;
    var text = el.textContent.trim();
    var m = text.match(/^(-?\d+(?:\.\d+)?)(.*)$/);
    if (!m) return;
    el.dataset.tbDone = "1";
    var target = parseFloat(m[1]);
    var suffix = m[2];
    var decimals = (m[1].split(".")[1] || "").length;
    var start = null, dur = 700;
    function step(ts) {
      if (!start) start = ts;
      var p = Math.min((ts - start) / dur, 1);
      var eased = 1 - Math.pow(1 - p, 3);
      el.textContent = (target * eased).toFixed(decimals) + suffix;
      if (p < 1) requestAnimationFrame(step);
      else el.textContent = text;
    }
    requestAnimationFrame(step);
  }

  /* 3. Fade-in reveal for cards, images and alerts */
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      if (en.isIntersecting) {
        en.target.classList.add("tb-in");
        io.unobserve(en.target);
      }
    });
  }, { threshold: 0.1 });

  function enhance() {
    doc.querySelectorAll('[data-testid="stMetricValue"]').forEach(countUp);
    doc.querySelectorAll(
      '[data-testid="stMetric"], [data-testid="stImage"], [data-testid="stAlert"], [data-testid="stChatMessage"]'
    ).forEach(function (el) {
      if (!el.classList.contains("tb-reveal")) {
        el.classList.add("tb-reveal");
        io.observe(el);
      }
    });
  }

  /* Streamlit re-renders often, so watch for changes (throttled) */
  var pending = false;
  new MutationObserver(function () {
    if (pending) return;
    pending = true;
    setTimeout(function () { pending = false; enhance(); }, 150);
  }).observe(doc.body, { childList: true, subtree: true });

  enhance();
})();
