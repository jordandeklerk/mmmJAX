/*
Sketches the landing page mark. The contour strokes draw in quick overlapping
succession, and then a coral pen writes the mmm. Instant navigation swaps in a
fresh mark, so each visit to the landing page replays it.
*/
(function () {
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
  if (!reduce.matches) {
    document.documentElement.classList.add("mmmj-motion");
  }

  function clamp(u) {
    return u < 0 ? 0 : u > 1 ? 1 : u;
  }

  // A pen stroke eases in and out without stopping dead at either end.
  function stroke(u) {
    return 0.6 * u + 0.4 * u * u * (3 - 2 * u);
  }

  function glide(u) {
    return 0.5 - 0.5 * Math.cos(Math.PI * u);
  }

  function play(mark) {
    if (!mark || reduce.matches || mark.classList.contains("is-live")) {
      return;
    }
    mark.classList.add("is-live");

    var sketches = Array.prototype.slice.call(mark.querySelectorAll(".mmmj-sketch"));
    var fill = mark.querySelector(".mmmj-sketch-fill");
    var ink = mark.querySelector(".mmmj-ink");
    var pen = mark.querySelector(".mmmj-pen");
    var first = mark.querySelector(".mmmj-ink-dot--start");
    var last = mark.querySelector(".mmmj-ink-dot--end");

    var lengths = sketches.map(function (path) {
      return path.getTotalLength();
    });
    var total = lengths.reduce(function (a, b) {
      return a + b;
    }, 0);

    // Each stroke starts once the strokes before it have covered their share of the contours.
    var drawn = 0;
    var plan = sketches.map(function (path, i) {
      var start = 0.1 + 1.3 * (drawn / total);
      drawn += lengths[i];
      return { path: path, start: start, end: start + Math.min(0.8, Math.max(0.28, lengths[i] / 2800)) };
    });
    var sketched = Math.max.apply(
      null,
      plan.map(function (step) {
        return step.end;
      }),
    );
    var write = { start: sketched - 0.2, end: sketched + 1.1 };
    var inkLength = ink.getTotalLength();
    var origin = null;

    function frame(now) {
      if (!mark.isConnected) {
        return;
      }
      if (origin === null) {
        origin = now;
      }
      var t = (now - origin) / 1000;

      plan.forEach(function (step) {
        var u = clamp((t - step.start) / (step.end - step.start));
        step.path.style.strokeDashoffset = u > 0 ? 1 - stroke(u) : 1.05;
      });
      if (t >= sketched) {
        fill.classList.add("is-drawn");
      }

      var w = clamp((t - write.start) / (write.end - write.start));
      if (t >= write.start - 0.1) {
        first.classList.add("is-drawn");
      }
      ink.style.strokeDashoffset = w > 0 ? 1 - glide(w) : 1.05;
      if (w > 0 && w < 1) {
        var point = ink.getPointAtLength(glide(w) * inkLength);
        pen.setAttribute("transform", "translate(" + point.x + " " + point.y + ")");
        pen.style.opacity = 1;
      }
      if (w >= 1) {
        last.classList.add("is-drawn");
        pen.style.opacity = 0;
        return;
      }
      window.requestAnimationFrame(frame);
    }

    window.requestAnimationFrame(frame);
  }

  function start() {
    play(document.querySelector(".mmmj-mark"));
  }

  if (document.readyState !== "loading") {
    start();
  } else {
    document.addEventListener("DOMContentLoaded", start);
  }
  if (typeof document$ !== "undefined" && document$ && document$.subscribe) {
    document$.subscribe(start);
  }
})();
