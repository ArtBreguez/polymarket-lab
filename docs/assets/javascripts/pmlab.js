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

  function reveal() {
    var items = document.querySelectorAll(".pm-reveal");
    if (!items.length) return;
    if (!("IntersectionObserver" in window)) {
      items.forEach(function (el) { el.classList.add("is-in"); });
      return;
    }
    var seen = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-in");
            seen.unobserve(entry.target);
          }
        });
      },
      { rootMargin: "0px 0px -8% 0px" }
    );
    items.forEach(function (el) { seen.observe(el); });
  }

  var box = document.querySelector("[data-install]");
  if (box) install(box);
  reveal();
})();
