(function () {
  const PRIMARY_SIGNAL_IDS = ["wt2", "wt1"];

  function normalizeTime(time) {
    if (time === undefined || time === null) {
      return null;
    }
    if (typeof time === "number" && Number.isFinite(time)) {
      return time;
    }
    if (typeof time === "string") {
      const parsed = Number(time);
      return Number.isFinite(parsed) ? parsed : null;
    }
    if (typeof time === "object") {
      if (typeof time.timestamp === "number") {
        return time.timestamp;
      }
      if (
        typeof time.year === "number" &&
        typeof time.month === "number" &&
        typeof time.day === "number"
      ) {
        return Math.floor(Date.UTC(time.year, time.month - 1, time.day) / 1000);
      }
    }
    return null;
  }

  function normalizeTradeNo(value) {
    if (value === undefined || value === null || value === "") {
      return null;
    }
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function baseChartOptions(container) {
    return {
      layout: {
        background: { color: "transparent" },
        textColor: "#dbeafe",
      },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.06)" },
        horzLines: { color: "rgba(255,255,255,0.06)" },
      },
      rightPriceScale: {
        borderColor: "rgba(255,255,255,0.12)",
      },
      timeScale: {
        borderColor: "rgba(255,255,255,0.12)",
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
        vertLine: { visible: false, labelVisible: false },
        horzLine: { visible: false, labelVisible: false },
      },
      width: container.clientWidth,
      height: container.clientHeight,
    };
  }

  function createChart(container, extraOptions) {
    const options = baseChartOptions(container);
    if (extraOptions) {
      if (extraOptions.layout) {
        options.layout = Object.assign({}, options.layout, extraOptions.layout);
      }
      if (extraOptions.grid) {
        options.grid = Object.assign({}, options.grid, extraOptions.grid);
      }
      if (extraOptions.rightPriceScale) {
        options.rightPriceScale = Object.assign({}, options.rightPriceScale, extraOptions.rightPriceScale);
      }
      if (extraOptions.timeScale) {
        options.timeScale = Object.assign({}, options.timeScale, extraOptions.timeScale);
      }
      if (extraOptions.crosshair) {
        options.crosshair = Object.assign({}, options.crosshair, extraOptions.crosshair);
      }
    }
    return LightweightCharts.createChart(container, options);
  }

  function hideGuide(element) {
    if (element) {
      element.style.display = "none";
    }
  }

  function showGuide(element, axis, value) {
    if (!element || !Number.isFinite(value)) {
      hideGuide(element);
      return;
    }
    element.style.display = "block";
    element.style.transform = axis === "x"
      ? "translateX(" + Math.round(value) + "px)"
      : "translateY(" + Math.round(value) + "px)";
  }

  function hideCrosshair(state) {
    if (!state) {
      return;
    }
    state.lastCrosshair = null;
    hideGuide(state.crosshairVertical);
    hideGuide(state.crosshairPrice);
    hideGuide(state.crosshairSignal);
  }

  function buildCandleLookup(candles) {
    const map = new Map();
    (candles || []).forEach(function (item) {
      const time = normalizeTime(item.time);
      if (time !== null) {
        map.set(time, item);
      }
    });
    return map;
  }

  function buildSignalLookup(lines) {
    let source = null;
    PRIMARY_SIGNAL_IDS.some(function (id) {
      source = (lines || []).find(function (line) {
        return line.id === id && Array.isArray(line.data) && line.data.length;
      });
      return Boolean(source);
    });
    if (!source) {
      source = (lines || []).find(function (line) {
        return Array.isArray(line.data) && line.data.length;
      }) || null;
    }

    const values = new Map();
    if (source) {
      source.data.forEach(function (point) {
        const time = normalizeTime(point.time);
        const value = Number(point.value);
        if (time !== null && Number.isFinite(value)) {
          values.set(time, value);
        }
      });
    }

    return {
      id: source ? source.id : null,
      values: values,
    };
  }

  function destroyChartState(state) {
    if (!state) {
      return;
    }
    if (state.wheelTargets && state.wheelHandler) {
      state.wheelTargets.forEach(function (target) {
        if (target) {
          target.removeEventListener("wheel", state.wheelHandler, false);
        }
      });
    }
    if (state.resizeObserver) {
      state.resizeObserver.disconnect();
    }
    if (state.overlayFrame) {
      cancelAnimationFrame(state.overlayFrame);
    }
    try {
      if (state.priceChart && typeof state.priceChart.remove === "function") {
        state.priceChart.remove();
      }
    } catch (error) {
      console.warn("Could not remove price chart", error);
    }
    try {
      if (state.signalChart && typeof state.signalChart.remove === "function") {
        state.signalChart.remove();
      }
    } catch (error) {
      console.warn("Could not remove signal chart", error);
    }
  }

  function applyChartSizes(state) {
    if (!state) {
      return;
    }
    if (state.priceHost.clientWidth && state.priceHost.clientHeight) {
      state.priceChart.applyOptions({
        width: state.priceHost.clientWidth,
        height: state.priceHost.clientHeight,
      });
    }
    if (state.signalHost.clientWidth && state.signalHost.clientHeight) {
      state.signalChart.applyOptions({
        width: state.signalHost.clientWidth,
        height: state.signalHost.clientHeight,
      });
    }
  }

  function renderTradePins(state) {
    if (!state || !state.pinLayer) {
      return;
    }
    const width = state.priceHost.clientWidth;
    const height = state.priceHost.clientHeight;
    const selectedTradeNo = normalizeTradeNo(state.selectedTradeNo);
    const html = [];

    (state.tradePins || []).forEach(function (pin) {
      const time = normalizeTime(pin.time);
      const price = Number(pin.price);
      const x = state.priceChart.timeScale().timeToCoordinate(time);
      const y = state.candles.priceToCoordinate(price);
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        return;
      }
      if (x < -40 || x > width + 40 || y < -60 || y > height + 60) {
        return;
      }

      const classes = ["tv-trade-pin"];
      classes.push(pin.anchor === "below" ? "is-below" : "is-above");
      classes.push(pin.kind === "entry" ? "is-entry" : "is-exit");
      if (pin.side === "short") {
        classes.push("is-short");
      }
      if (selectedTradeNo !== null && normalizeTradeNo(pin.tradeNo) === selectedTradeNo) {
        classes.push("is-selected");
      }

      html.push(
        '<div class="' + classes.join(" ") + '" style="left:' + Math.round(x) + "px;top:" +
        Math.round(y) + 'px;--pin-color:' + escapeHtml(pin.color || "#69b7ff") +
        ';" title="' + escapeHtml(pin.tooltip || pin.label || "") + '">' +
        "<span>" + escapeHtml(pin.label || "") + "</span></div>"
      );
    });

    state.pinLayer.innerHTML = html.join("");
  }

  function getPriceCrosshairY(state, time) {
    const candle = state.candleLookup.get(time);
    if (!candle) {
      return null;
    }
    const close = Number(candle.close);
    return Number.isFinite(close) ? state.candles.priceToCoordinate(close) : null;
  }

  function getSignalCrosshairY(state, time) {
    if (!state.signalLookup.id) {
      return null;
    }
    const series = state.signalLineSeries[state.signalLookup.id];
    const value = state.signalLookup.values.get(time);
    if (!series || !Number.isFinite(value) || typeof series.priceToCoordinate !== "function") {
      return null;
    }
    return series.priceToCoordinate(value);
  }

  function drawCrosshair(state, sourceKey, param) {
    if (!state || !param || !param.point) {
      hideCrosshair(state);
      return;
    }

    const time = normalizeTime(param.time);
    if (time === null) {
      hideCrosshair(state);
      return;
    }

    const sourcePane = sourceKey === "signal" ? state.signalPane : state.pricePane;
    const sourceHeight = sourceKey === "signal" ? state.signalHost.clientHeight : state.priceHost.clientHeight;
    if (
      !Number.isFinite(param.point.x) ||
      !Number.isFinite(param.point.y) ||
      param.point.x < 0 ||
      param.point.y < 0 ||
      param.point.x > sourcePane.clientWidth ||
      param.point.y > sourceHeight
    ) {
      hideCrosshair(state);
      return;
    }

    const x = sourcePane.offsetLeft + param.point.x;
    const priceY = sourceKey === "price" ? param.point.y : getPriceCrosshairY(state, time);
    const signalY = sourceKey === "signal" ? param.point.y : getSignalCrosshairY(state, time);

    state.lastCrosshair = { sourceKey: sourceKey, time: time, point: { x: param.point.x, y: param.point.y } };
    showGuide(state.crosshairVertical, "x", x);
    showGuide(state.crosshairPrice, "y", priceY);
    showGuide(state.crosshairSignal, "y", signalY);
  }

  function refreshOverlays(state) {
    if (!state) {
      return;
    }
    state.overlayFrame = 0;
    renderTradePins(state);
    if (state.lastCrosshair) {
      drawCrosshair(state, state.lastCrosshair.sourceKey, state.lastCrosshair);
    } else {
      hideGuide(state.crosshairVertical);
      hideGuide(state.crosshairPrice);
      hideGuide(state.crosshairSignal);
    }
  }

  function scheduleOverlayRefresh(state) {
    if (!state || state.overlayFrame) {
      return;
    }
    state.overlayFrame = requestAnimationFrame(function () {
      refreshOverlays(state);
    });
  }

  function linkTimeScales(state) {
    let syncing = false;

    function mirror(source, target) {
      if (
        !source ||
        !target ||
        !source.timeScale ||
        !target.timeScale ||
        typeof source.timeScale().subscribeVisibleTimeRangeChange !== "function"
      ) {
        return;
      }
      source.timeScale().subscribeVisibleTimeRangeChange(function (range) {
        if (syncing || !range) {
          scheduleOverlayRefresh(state);
          return;
        }
        syncing = true;
        try {
          target.timeScale().setVisibleRange(range);
        } catch (error) {
          console.warn("Could not sync chart time range", error);
        }
        syncing = false;
        scheduleOverlayRefresh(state);
      });
    }

    mirror(state.priceChart, state.signalChart);
    mirror(state.signalChart, state.priceChart);
  }

  function subscribeCrosshair(state) {
    state.priceChart.subscribeCrosshairMove(function (param) {
      if (!param || param.time === undefined) {
        hideCrosshair(state);
        return;
      }
      drawCrosshair(state, "price", param);
    });

    state.signalChart.subscribeCrosshairMove(function (param) {
      if (!param || param.time === undefined) {
        hideCrosshair(state);
        return;
      }
      drawCrosshair(state, "signal", param);
    });

    ["mouseleave", "pointerleave"].forEach(function (eventName) {
      state.pricePane.addEventListener(eventName, function () {
        hideCrosshair(state);
      });
      state.signalPane.addEventListener(eventName, function () {
        hideCrosshair(state);
      });
    });
  }

  function bindWheelLock(state) {
    const wheelHandler = function (event) {
      if (!event) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
    };

    const wheelTargets = [state.container, state.pricePane, state.signalPane, state.priceHost, state.signalHost];
    wheelTargets.forEach(function (target) {
      if (target) {
        target.addEventListener("wheel", wheelHandler, { passive: false });
      }
    });

    state.wheelHandler = wheelHandler;
    state.wheelTargets = wheelTargets;
  }

  function buildChartState(container) {
    container.innerHTML = [
      '<div class="tv-crosshair-v"></div>',
      '<div class="tv-chart-pane tv-chart-price">',
      '  <div class="tv-chart-host tv-chart-price-host"></div>',
      '  <div class="tv-chart-pane-overlay tv-chart-price-overlay">',
      '    <div class="tv-crosshair-h tv-crosshair-h-price"></div>',
      '    <div class="tv-trade-pins"></div>',
      "  </div>",
      "</div>",
      '<div class="tv-chart-pane tv-chart-signal">',
      '  <div class="tv-chart-host tv-chart-signal-host"></div>',
      '  <div class="tv-chart-pane-overlay tv-chart-signal-overlay">',
      '    <div class="tv-crosshair-h tv-crosshair-h-signal"></div>',
      "  </div>",
      "</div>",
    ].join("");

    const pricePane = container.querySelector(".tv-chart-price");
    const signalPane = container.querySelector(".tv-chart-signal");
    const priceHost = container.querySelector(".tv-chart-price-host");
    const signalHost = container.querySelector(".tv-chart-signal-host");

    const priceChart = createChart(priceHost);
    const signalChart = createChart(signalHost, {
      rightPriceScale: {
        scaleMargins: { top: 0.08, bottom: 0.08 },
      },
    });

    const candles = priceChart.addSeries(LightweightCharts.CandlestickSeries, {
      upColor: "#22d3aa",
      downColor: "#ff7a59",
      wickUpColor: "#22d3aa",
      wickDownColor: "#ff7a59",
      borderVisible: false,
    });

    const state = {
      container: container,
      pricePane: pricePane,
      signalPane: signalPane,
      priceHost: priceHost,
      signalHost: signalHost,
      priceChart: priceChart,
      signalChart: signalChart,
      candles: candles,
      priceLineSeries: {},
      signalLineSeries: {},
      markersApi: null,
      resizeObserver: null,
      crosshairVertical: container.querySelector(".tv-crosshair-v"),
      crosshairPrice: container.querySelector(".tv-crosshair-h-price"),
      crosshairSignal: container.querySelector(".tv-crosshair-h-signal"),
      pinLayer: container.querySelector(".tv-trade-pins"),
      tradePins: [],
      candleLookup: new Map(),
      signalLookup: { id: null, values: new Map() },
      selectedTradeNo: null,
      lastCrosshair: null,
      overlayFrame: 0,
      wheelHandler: null,
      wheelTargets: [],
    };

    linkTimeScales(state);
    subscribeCrosshair(state);
    bindWheelLock(state);

    const resizeObserver = new ResizeObserver(function () {
      applyChartSizes(state);
      scheduleOverlayRefresh(state);
    });
    state.resizeObserver = resizeObserver;
    resizeObserver.observe(container);
    resizeObserver.observe(pricePane);
    resizeObserver.observe(signalPane);

    applyChartSizes(state);
    return state;
  }

  function ensureChart(container) {
    const current = window.__bee7LightweightChart;
    if (current && current.container === container) {
      return current;
    }

    if (current) {
      destroyChartState(current);
    }

    window.__bee7LightweightChart = buildChartState(container);
    return window.__bee7LightweightChart;
  }

  function lineSeriesOptions(line) {
    const options = {
      color: line.color,
      lineWidth: line.lineWidth || 2,
      visible: line.visible !== false,
      priceLineVisible: line.priceLineVisible === true,
      lastValueVisible: line.lastValueVisible === true,
      crosshairMarkerVisible: line.crosshairMarkerVisible === true,
    };
    if (line.lineStyle !== undefined) {
      options.lineStyle = line.lineStyle;
    }
    return options;
  }

  function showRenderError(container, error) {
    console.error("Bee1 chart render failed", error);
    destroyChartState(window.__bee7LightweightChart);
    window.__bee7LightweightChart = null;
    container.innerHTML = [
      '<div style="padding:18px;color:#ffb454;background:rgba(255,180,84,0.08);border:1px solid rgba(255,180,84,0.22);border-radius:18px;">',
      "<strong>Nie udalo sie wyrenderowac wykresu.</strong><br>",
      '<span style="color:#95a6bc;">Odswiez strone przez Ctrl+F5. Jezeli blad wroci, renderer pokaze szczegoly w konsoli przegladarki.</span>',
      "</div>",
    ].join("");
  }

  function syncLineSeries(chart, seriesStore, lines) {
    const activeIds = new Set();

    (lines || []).forEach(function (line) {
      let series = seriesStore[line.id];
      if (!series) {
        series = chart.addSeries(LightweightCharts.LineSeries, lineSeriesOptions(line));
        seriesStore[line.id] = series;
      }

      series.applyOptions(lineSeriesOptions(line));
      series.setData(line.data || []);
      activeIds.add(line.id);
    });

    Object.keys(seriesStore).forEach(function (id) {
      if (!activeIds.has(id)) {
        seriesStore[id].setData([]);
        seriesStore[id].applyOptions({ visible: false });
      }
    });
  }

  function render(payload) {
    const attempt = arguments.length > 1 ? arguments[1] : 0;
    const container = document.getElementById("tv-chart");
    if (!container) {
      if (payload && attempt < 20) {
        setTimeout(function () {
          render(payload, attempt + 1);
        }, 80);
      }
      return "";
    }

    if (typeof LightweightCharts === "undefined") {
      if (attempt < 20) {
        setTimeout(function () {
          render(payload, attempt + 1);
        }, 80);
      } else {
        container.innerHTML = '<div style="padding:16px;color:#95a6bc;">Lightweight Charts failed to load.</div>';
      }
      return "";
    }

    if (!container.clientWidth || !container.clientHeight) {
      if (attempt < 20) {
        setTimeout(function () {
          render(payload, attempt + 1);
        }, 80);
      }
      return "";
    }

    try {
      const state = ensureChart(container);
      if (
        !state.priceHost.clientWidth ||
        !state.priceHost.clientHeight ||
        !state.signalHost.clientWidth ||
        !state.signalHost.clientHeight
      ) {
        if (attempt < 20) {
          setTimeout(function () {
            render(payload, attempt + 1);
          }, 80);
        }
        return "";
      }

      state.candles.setData(payload.candles || []);
      syncLineSeries(state.priceChart, state.priceLineSeries, payload.lines || []);
      syncLineSeries(state.signalChart, state.signalLineSeries, payload.signalLines || []);

      state.candleLookup = buildCandleLookup(payload.candles || []);
      state.signalLookup = buildSignalLookup(payload.signalLines || []);
      state.tradePins = payload.tradePins || [];
      state.selectedTradeNo = payload.selectedTradeNo;

      if (state.markersApi && typeof state.markersApi.setMarkers === "function") {
        state.markersApi.setMarkers([]);
      }
      state.markersApi = LightweightCharts.createSeriesMarkers(state.candles, payload.markers || []);

      applyChartSizes(state);

      if (payload.focusRange && payload.focusRange.from && payload.focusRange.to) {
        state.priceChart.timeScale().setVisibleRange(payload.focusRange);
        state.signalChart.timeScale().setVisibleRange(payload.focusRange);
      } else {
        state.priceChart.timeScale().fitContent();
        state.signalChart.timeScale().fitContent();
      }

      scheduleOverlayRefresh(state);
    } catch (error) {
      if (attempt < 1) {
        destroyChartState(window.__bee7LightweightChart);
        window.__bee7LightweightChart = null;
        setTimeout(function () {
          render(payload, attempt + 1);
        }, 80);
      } else {
        showRenderError(container, error);
      }
      return "";
    }

    return [payload.symbol || "", payload.tf || "", Date.now()].join(":");
  }

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    bee7_dashboard: {
      render: render,
    },
  });
})();

