(function () {
  if (window.__moaBridgeLoaded) return;
  window.__moaBridgeLoaded = true;

  // Only run inside the llama.cpp WebUI, not on random pages
  if (location.hostname !== "127.0.0.1") return;

  function checkAndInject() {
    browser.storage.local.get("moaPendingText").then(function (result) {
      var text = (result && result.moaPendingText) || "";
      if (!text) return;

      browser.storage.local.remove("moaPendingText");
      waitAndInject(text, 0);
    }).catch(function () {});
  }

  function waitAndInject(text, attempts) {
    if (attempts > 50) return;

    var textarea = document.querySelector("textarea");
    if (!textarea) {
      setTimeout(function () { waitAndInject(text, attempts + 1); }, 200);
      return;
    }

    var setter = Object.getOwnPropertyDescriptor(
      HTMLTextAreaElement.prototype, "value"
    ).set;
    setter.call(textarea, text);
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
    textarea.focus();
  }

  // Check immediately on load
  checkAndInject();

  // Listen for future actions while sidebar is already open
  browser.storage.onChanged.addListener(function (changes, area) {
    if (area === "local" && changes.moaPendingText && changes.moaPendingText.newValue) {
      checkAndInject();
    }
  });
})();