/*
Sketches the landing page mark. A clockwise front starts each contour stroke as
it passes the stroke's first point, every stroke draws at one shared speed that
eases in and out, and then a coral pen writes the mmm. The template's strokes
all turn clockwise and are cut to similar lengths, so no pen races another.
Instant navigation swaps in a fresh mark, so each visit to the landing page
replays it.
*/
(function () {
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
  if (!reduce.matches) {
    document.documentElement.classList.add("mmmj-motion");
  }

  function clamp(u) {
    return u < 0 ? 0 : u > 1 ? 1 : u;
  }

  // Mostly a cosine ease, with some linear motion so the sketch never sits still at either end.
  function warp(u) {
    return 0.3 * u + 0.7 * (0.5 - 0.5 * Math.cos(Math.PI * u));
  }

  function glide(u) {
    return 0.5 - 0.5 * Math.cos(Math.PI * u);
  }

  // The share of the sketch's duration at which the eased progress reaches tau.
  function unwarp(tau) {
    var low = 0;
    var high = 1;
    for (var i = 0; i < 30; i++) {
      var middle = (low + high) / 2;
      if (warp(middle) < tau) {
        low = middle;
      } else {
        high = middle;
      }
    }
    return high;
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

    var view = mark.viewBox.baseVal;
    var centerX = view.x + view.width / 2;
    var centerY = view.y + view.height / 2;
    var lengths = sketches.map(function (path) {
      return path.getTotalLength();
    });
    var longest = Math.max.apply(null, lengths);

    // The front finishes its turn by this share of the sketch, leaving the rest for the strokes it starts last.
    var turn = 0.55;
    var speed = longest / (1 - turn);
    var plan = sketches.map(function (path, i) {
      var point = path.getPointAtLength(0);
      var angle = Math.atan2(point.y - centerY, point.x - centerX) + Math.PI / 2;
      var clockwise = (angle + 2 * Math.PI) % (2 * Math.PI);
      return { path: path, start: (turn * clockwise) / (2 * Math.PI), span: lengths[i] / speed, shown: -1 };
    });

    var lead = 0.05;
    var duration = 2.3;
    var done = Math.max.apply(
      null,
      plan.map(function (step) {
        return step.start + step.span;
      }),
    );
    var sketched = lead + duration * unwarp(done);
    var write = { start: sketched - 0.2, end: sketched + 1.1 };
    var inkLength = ink.getTotalLength();
    var clock = 0;
    var previous = null;

    function frame(now) {
      if (!mark.isConnected) {
        return;
      }
      // A slow frame moves the clock at most a thirtieth of a second, so the sketch pauses instead of jumping.
      if (previous !== null) {
        clock += Math.min(now - previous, 1000 / 30) / 1000;
      }
      previous = now;

      var progress = warp(clamp((clock - lead) / duration));
      plan.forEach(function (step) {
        var u = clamp((progress - step.start) / step.span);
        if (u !== step.shown) {
          step.shown = u;
          step.path.style.strokeDashoffset = u > 0 ? 1 - u : 1.05;
        }
      });
      if (clock >= sketched) {
        fill.classList.add("is-drawn");
      }

      var w = clamp((clock - write.start) / (write.end - write.start));
      if (clock >= write.start - 0.1) {
        first.classList.add("is-drawn");
      }
      if (w > 0) {
        ink.style.strokeDashoffset = 1 - glide(w);
      }
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
