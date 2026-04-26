(function () {
  function buildChartState(container) {
    const chart = LightweightCharts.createChart(container, {
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
      },
    });

    const candles = chart.addSeries(LightweightCharts.CandlestickSeries, {
      upColor: "#22d3aa",
      downColor: "#ff7a59",
      wickUpColor: "#22d3aa",
      wickDownColor: "#ff7a59",
      borderVisible: false,
    });

    const resizeObserver = new ResizeObserver(function () {
      chart.applyOptions({
        width: container.clientWidth,
        height: container.clientHeight,
      });
    });
    resizeObserver.observe(container);

    chart.applyOptions({
      width: container.clientWidth,
      height: container.clientHeight,
    });

    return {
      container: container,
      chart: chart,
      candles: candles,
      lineSeries: {},
      markersApi: null,
      resizeObserver: resizeObserver,
    };
  }

  function ensureChart(container) {
    const current = window.__bee7LightweightChart;
    if (current && current.container === container) {
      return current;
    }

    if (current && current.resizeObserver) {
      current.resizeObserver.disconnect();
    }

    container.innerHTML = "";
    window.__bee7LightweightChart = buildChartState(container);
    return window.__bee7LightweightChart;
  }

  function syncLineSeries(state, lines) {
    const activeIds = new Set();

    (lines || []).forEach(function (line) {
      let series = state.lineSeries[line.id];
      if (!series) {
        series = state.chart.addSeries(LightweightCharts.LineSeries, {
          color: line.color,
          lineWidth: line.lineWidth || 2,
          visible: line.visible !== false,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        });
        state.lineSeries[line.id] = series;
      }

      series.applyOptions({
        color: line.color,
        lineWidth: line.lineWidth || 2,
        visible: line.visible !== false,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      });
      series.setData(line.data || []);
      activeIds.add(line.id);
    });

    Object.keys(state.lineSeries).forEach(function (id) {
      if (!activeIds.has(id)) {
        state.lineSeries[id].setData([]);
        state.lineSeries[id].applyOptions({ visible: false });
      }
    });
  }

  function render(payload) {
    const container = document.getElementById("tv-chart");
    if (!container) {
      return "";
    }
    if (typeof LightweightCharts === "undefined") {
      container.innerHTML = '<div style="padding:16px;color:#95a6bc;">Lightweight Charts failed to load.</div>';
      return "";
    }

    const state = ensureChart(container);
    state.candles.setData(payload.candles || []);
    syncLineSeries(state, payload.lines || []);

    if (state.markersApi && typeof state.markersApi.setMarkers === "function") {
      state.markersApi.setMarkers([]);
    }
    state.markersApi = LightweightCharts.createSeriesMarkers(state.candles, payload.markers || []);

    state.chart.applyOptions({
      width: container.clientWidth,
      height: container.clientHeight,
    });

    if (payload.focusRange && payload.focusRange.from && payload.focusRange.to) {
      state.chart.timeScale().setVisibleRange(payload.focusRange);
    } else {
      state.chart.timeScale().fitContent();
    }

    return [payload.symbol || "", payload.tf || "", Date.now()].join(":");
  }

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    bee7_dashboard: {
      render: render,
    },
  });
})();

