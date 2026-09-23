/* Make the header title link to the documentation home page. */
(function () {
  function linkTitle() {
    var logo = document.querySelector(".md-header__button.md-logo");
    var topic = document.querySelector(".md-header__title .md-header__topic .md-ellipsis");
    if (!logo || !topic || topic.closest("a")) {
      return;
    }
    var anchor = document.createElement("a");
    anchor.href = logo.getAttribute("href");
    anchor.style.color = "inherit";
    anchor.style.textDecoration = "none";
    topic.parentNode.insertBefore(anchor, topic);
    anchor.appendChild(topic);
  }

  if (document.readyState !== "loading") {
    linkTitle();
  } else {
    document.addEventListener("DOMContentLoaded", linkTitle);
  }
  if (typeof document$ !== "undefined" && document$ && document$.subscribe) {
    document$.subscribe(linkTitle);
  }
})();
