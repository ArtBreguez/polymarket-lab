/* pmlab docs: the only script the theme adds (Material's own bundle handles search and navigation).
   Install tabs with a copy button, and reveal-on-scroll for the home page. No dependency. */
(function () {
  "use strict";

  function copy(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      var area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.cssText = "position:fixed;top:0;opacity:0";
      document.body.appendChild(area);
      area.select();
      var ok = false;
      try {
        ok = document.execCommand("copy");
      } catch (e) {
        ok = false;
      }
      document.body.removeChild(area);
      ok ? resolve() : reject(new Error("copy failed"));
    });
  }

  function install(root) {
    var cmd = root.querySelector("[data-install-cmd]");
    var button = root.querySelector("[data-install-copy]");
    var tabs = root.querySelectorAll("[data-install-tab]");
    var live = root.querySelector("[data-install-live]");
    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        tabs.forEach(function (other) {
          other.setAttribute("aria-selected", String(other === tab));
        });
        cmd.textContent = tab.getAttribute("data-install-tab");
      });
    });
    var timer;
    button.addEventListener("click", function () {
      copy(cmd.textContent.trim()).then(
        function () {
          button.textContent = "copied";
          live.textContent = "Copied: " + cmd.textContent.trim();
        },
        function () {
          button.textContent = "press Ctrl+C";
        }
      );
      clearTimeout(timer);
      timer = setTimeout(function () {
        button.textContent = "copy";
        live.textContent = "";
      }, 1800);
    });
  }

  // Reveal on scroll: an element appears once its top is inside the upper 92% of the viewport, so a
  // jump past it (End, a search hit, an anchor) reveals it too: nothing can stay hidden.
  function reveal() {
    var pending = Array.prototype.slice.call(document.querySelectorAll(".pm-reveal"));
    if (!pending.length) return;
    var queued = false;
    function check() {
      queued = false;
      var line = window.innerHeight * 0.92;
      pending = pending.filter(function (el) {
        if (el.getBoundingClientRect().top < line) {
          el.classList.add("is-in");
          return false;
        }
        return true;
      });
      if (!pending.length) {
        window.removeEventListener("scroll", onScroll);
        window.removeEventListener("resize", onScroll);
      }
    }
    function onScroll() {
      if (!queued) {
        queued = true;
        window.requestAnimationFrame(check);
      }
    }
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    check();
  }

  var box = document.querySelector("[data-install]");
  if (box) install(box);
  reveal();
})();
