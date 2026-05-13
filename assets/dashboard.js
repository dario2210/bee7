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

  function safeClass(value) {
    return String(value || "")
      .replace(/[^a-z0-9_-]/gi, "-")
      .toLowerCase();
  }

  function baseChartOptions(container) {
    return {
      layout: {
        background: { color: "transparent" },
        textColor: "#dbeafe",
        panes: {
          separatorColor: "rgba(131,153,179,0.22)",
          separatorHoverColor: "rgba(105,183,255,0.62)",
          enableResize: false,
        },
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
        fixLeftEdge: true,
        fixRightEdge: true,
        lockVisibleTimeRangeOnResize: true,
        rightBarStaysOnScroll: true,
        rightOffset: 4,
        minBarSpacing: 0.45,
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: false,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
      crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
      },
      width: container.clientWidth,
      height: container.clientHeight,
    };
  }

  function normalizeRange(range) {
    if (!range) {
      return null;
    }
    const from = normalizeTime(range.from);
    const to = normalizeTime(range.to);
    if (from === null || to === null || !Number.isFinite(from) || !Number.isFinite(to)) {
      return null;
    }
    return to < from ? { from: to, to: from } : { from: from, to: to };
  }

  function buildDataBounds(candles) {
    const times = (candles || [])
      .map(function (item) { return normalizeTime(item.time); })
      .filter(function (time) { return time !== null && Number.isFinite(time); });
    if (!times.length) {
      return null;
    }
    return {
      from: Math.min.apply(null, times),
      to: Math.max.apply(null, times),
    };
  }

  function clampRangeToData(state, range) {
    const normalized = normalizeRange(range);
    const bounds = state ? state.dataBounds : null;
    if (!normalized || !bounds) {
      return normalized;
    }

    const fullSpan = Math.max(bounds.to - bounds.from, 1);
    const span = Math.max(normalized.to - normalized.from, Math.min(fullSpan, 60));
    if (span >= fullSpan) {
      return { from: bounds.from, to: bounds.to };
    }

    let from = normalized.from;
    let to = normalized.to;
    if (from < bounds.from) {
      from = bounds.from;
      to = from + span;
    }
    if (to > bounds.to) {
      to = bounds.to;
      from = to - span;
    }
    return {
      from: Math.max(from, bounds.from),
      to: Math.min(to, bounds.to),
    };
  }

  function payloadKey(payload) {
    const candles = payload.candles || [];
    const first = candles.length ? normalizeTime(candles[0].time) : "";
    const last = candles.length ? normalizeTime(candles[candles.length - 1].time) : "";
    return [
      payload.symbol || "",
      payload.tf || "",
      payload.chartView || "",
      candles.length,
      first,
      last,
      (payload.lines || []).length,
      (payload.signalLines || []).length,
      (payload.markers || []).length,
      (payload.signalMarkers || []).length,
      (payload.tradePins || []).length,
      payload.selectedTradeNo || "",
    ].join("|");
  }

  function focusKey(payload) {
    const range = normalizeRange(payload.focusRange);
    return range ? [range.from, range.to].join(":") : "";
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

  function syncLineSeries(chart, seriesStore, lines, paneIndex) {
    const activeIds = new Set();

    (lines || []).forEach(function (line) {
      let series = seriesStore[line.id];
      if (!series) {
        series = chart.addSeries(LightweightCharts.LineSeries, lineSeriesOptions(line), paneIndex);
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

  function syncSeriesMarkers(seriesStore, markerStore, markers) {
    const grouped = {};

    (markers || []).forEach(function (marker) {
      if (!marker || !marker.seriesId) {
        return;
      }
      if (!grouped[marker.seriesId]) {
        grouped[marker.seriesId] = [];
      }
      grouped[marker.seriesId].push({
        time: marker.time,
        position: marker.position || "aboveBar",
        shape: marker.shape || "circle",
        color: marker.color || "#69b7ff",
        text: marker.text || "",
      });
    });

    Object.keys(markerStore).forEach(function (seriesId) {
      if (!grouped[seriesId] && markerStore[seriesId] && typeof markerStore[seriesId].setMarkers === "function") {
        markerStore[seriesId].setMarkers([]);
      }
    });

    Object.keys(grouped).forEach(function (seriesId) {
      const series = seriesStore[seriesId];
      if (!series) {
        return;
      }
      if (markerStore[seriesId] && typeof markerStore[seriesId].setMarkers === "function") {
        markerStore[seriesId].setMarkers(grouped[seriesId]);
      } else {
        markerStore[seriesId] = LightweightCharts.createSeriesMarkers(series, grouped[seriesId]);
      }
    });
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

  function resizeChart(state) {
    if (!state || !state.host.clientWidth || !state.host.clientHeight) {
      return;
    }
    state.chart.resize(state.host.clientWidth, state.host.clientHeight);
    const panes = typeof state.chart.panes === "function" ? state.chart.panes() : [];
    if (panes && panes.length >= 2) {
      const paneHeight = Math.max(140, Math.floor((state.host.clientHeight - 42) / 2));
      try {
        panes[0].setHeight(paneHeight);
        panes[1].setHeight(paneHeight);
      } catch (error) {
        console.warn("Could not resize panes", error);
      }
    }
    scheduleOverlayRefresh(state);
  }

  function resetChartView(state) {
    if (!state) {
      return;
    }
    try {
      state.chart.timeScale().fitContent();
    } catch (error) {
      console.warn("Could not reset chart view", error);
    }
    scheduleOverlayRefresh(state);
  }

  function setChartRange(state, range) {
    const clamped = clampRangeToData(state, range);
    if (!state || !clamped) {
      return;
    }
    try {
      state.chart.timeScale().setVisibleRange(clamped);
    } catch (error) {
      console.warn("Could not apply chart range", error);
    }
    scheduleOverlayRefresh(state);
  }

  function renderTradePins(state) {
    if (!state || !state.pinLayer) {
      return;
    }
    const width = state.host.clientWidth;
    const height = state.host.clientHeight;
    const selectedTradeNo = normalizeTradeNo(state.selectedTradeNo);
    const html = [];

    (state.tradePins || []).forEach(function (pin) {
      const time = normalizeTime(pin.time);
      const price = Number(pin.price);
      const x = state.chart.timeScale().timeToCoordinate(time);
      const y = state.candles.priceToCoordinate(price);
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        return;
      }
      if (x < -50 || x > width + 50 || y < -80 || y > height + 80) {
        return;
      }

      const kind = safeClass(pin.kind || "entry") || "entry";
      const decision = safeClass(pin.decision || "");
      const rejectCode = safeClass(pin.rejectCode || "");
      const classes = ["tv-trade-pin"];
      classes.push(pin.anchor === "below" ? "is-below" : "is-above");
      classes.push("is-" + kind);
      if (decision) {
        classes.push("is-" + decision);
      }
      if (rejectCode) {
        classes.push("reject-" + rejectCode);
      }
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

  function refreshOverlays(state) {
    if (!state) {
      return;
    }
    state.overlayFrame = 0;
    renderTradePins(state);
  }

  function scheduleOverlayRefresh(state) {
    if (!state || state.overlayFrame) {
      return;
    }
    state.overlayFrame = requestAnimationFrame(function () {
      refreshOverlays(state);
    });
  }

  function destroyChartState(state) {
    if (!state) {
      return;
    }
    if (state.resizeObserver) {
      state.resizeObserver.disconnect();
    }
    if (state.overlayFrame) {
      cancelAnimationFrame(state.overlayFrame);
    }
    if (state.resetButton && state.resetHandler) {
      state.resetButton.removeEventListener("click", state.resetHandler);
    }
    try {
      if (state.chart && typeof state.chart.remove === "function") {
        state.chart.remove();
      }
    } catch (error) {
      console.warn("Could not remove chart", error);
    }
  }

  function buildChartState(container) {
    container.innerHTML = [
      '<button type="button" class="tv-chart-reset" title="Wroc do pelnego zakresu danych">Reset widoku</button>',
      '<div class="tv-chart-host tv-chart-main-host"></div>',
      '<div class="tv-chart-pane-overlay tv-chart-main-overlay">',
      '  <div class="tv-trade-pins"></div>',
      "</div>",
    ].join("");

    const host = container.querySelector(".tv-chart-main-host");
    const chart = LightweightCharts.createChart(host, baseChartOptions(host));
    const candles = chart.addSeries(LightweightCharts.CandlestickSeries, {
      upColor: "#22d3aa",
      downColor: "#ff7a59",
      wickUpColor: "#22d3aa",
      wickDownColor: "#ff7a59",
      borderVisible: false,
    }, 0);

    const state = {
      container: container,
      host: host,
      chart: chart,
      candles: candles,
      priceLineSeries: {},
      signalLineSeries: {},
      markersApi: null,
      signalMarkersApis: {},
      resizeObserver: null,
      resetButton: container.querySelector(".tv-chart-reset"),
      resetHandler: null,
      pinLayer: container.querySelector(".tv-trade-pins"),
      tradePins: [],
      candleLookup: new Map(),
      signalLookup: { id: null, values: new Map() },
      selectedTradeNo: null,
      dataBounds: null,
      lastPayloadKey: "",
      lastFocusKey: "",
      overlayFrame: 0,
    };

    state.resetHandler = function () {
      state.lastFocusKey = "";
      resetChartView(state);
    };
    if (state.resetButton) {
      state.resetButton.addEventListener("click", state.resetHandler);
    }

    if (chart.timeScale && typeof chart.timeScale().subscribeVisibleTimeRangeChange === "function") {
      chart.timeScale().subscribeVisibleTimeRangeChange(function () {
        scheduleOverlayRefresh(state);
      });
    }
    if (chart.subscribeCrosshairMove) {
      chart.subscribeCrosshairMove(function () {
        scheduleOverlayRefresh(state);
      });
    }

    const resizeObserver = new ResizeObserver(function () {
      resizeChart(state);
    });
    state.resizeObserver = resizeObserver;
    resizeObserver.observe(container);
    resizeObserver.observe(host);

    resizeChart(state);
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

  function showRenderError(container, error) {
    console.error("Bee7 chart render failed", error);
    destroyChartState(window.__bee7LightweightChart);
    window.__bee7LightweightChart = null;
    container.innerHTML = [
      '<div style="padding:18px;color:#ffb454;background:rgba(255,180,84,0.08);border:1px solid rgba(255,180,84,0.22);border-radius:18px;">',
      "<strong>Nie udalo sie wyrenderowac wykresu.</strong><br>",
      '<span style="color:#95a6bc;">Odswiez strone przez Ctrl+F5. Jezeli blad wroci, renderer pokaze szczegoly w konsoli przegladarki.</span>',
      "</div>",
    ].join("");
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
      if (!state.host.clientWidth || !state.host.clientHeight) {
        if (attempt < 20) {
          setTimeout(function () {
            render(payload, attempt + 1);
          }, 80);
        }
        return "";
      }

      const candlesData = payload.candles || [];
      const nextPayloadKey = payloadKey(payload);
      const nextFocusKey = focusKey(payload);
      const isNewPayload = state.lastPayloadKey !== nextPayloadKey;
      const isNewFocus = nextFocusKey && state.lastFocusKey !== nextFocusKey;

      state.dataBounds = buildDataBounds(candlesData);
      state.candles.setData(candlesData);
      syncLineSeries(state.chart, state.priceLineSeries, payload.lines || [], 0);
      syncLineSeries(state.chart, state.signalLineSeries, payload.signalLines || [], 1);

      state.candleLookup = buildCandleLookup(candlesData);
      state.signalLookup = buildSignalLookup(payload.signalLines || []);
      state.tradePins = payload.tradePins || [];
      state.selectedTradeNo = payload.selectedTradeNo;

      if (state.markersApi && typeof state.markersApi.setMarkers === "function") {
        state.markersApi.setMarkers([]);
      }
      state.markersApi = LightweightCharts.createSeriesMarkers(state.candles, payload.markers || []);
      syncSeriesMarkers(state.signalLineSeries, state.signalMarkersApis, payload.signalMarkers || []);

      resizeChart(state);

      if (nextFocusKey && payload.focusRange) {
        if (isNewFocus || isNewPayload) {
          setChartRange(state, payload.focusRange);
        }
      } else if (isNewPayload && !state.lastPayloadKey) {
        resetChartView(state);
      }

      state.lastPayloadKey = nextPayloadKey;
      state.lastFocusKey = nextFocusKey;
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

    return [payload.symbol || "", payload.tf || "", payload.chartView || "", Date.now()].join(":");
  }

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    bee7_dashboard: {
      render: render,
    },
  });
})();
