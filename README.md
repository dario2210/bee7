# bee7

`bee7` to pierwsza wersja strategii WaveTrend zbudowana na tej samej strukturze projektu co `bee1`.
Dashboard, wykres, flow backtest/WFO/live oraz uklad plikow pozostaja spojne z wczesniejszym projektem, ale logika wejsc i wyjsc zostala przestawiona na sygnaly WaveTrend.

## Co robi projekt

- domyslnie pracuje w trybie `long/short` z lustrzana logika po obu stronach rynku
- otwiera `long` na zielonej kropce WaveTrend pod zerem oraz w oknie kilku barow po tym sygnale
- otwiera `short` na czerwonej kropce WaveTrend nad zerem oraz w analogicznym oknie kilku barow po tym sygnale
- dla `long` wymaga odzyskania wybranej EMA, a dla `short` odrzucenia wybranej EMA, zeby odsiac slabsze setupy
- dopuszcza re-entry, gdy WaveTrend nadal utrzymuje sie blisko zera po ostatniej zielonej lub czerwonej kropce
- pozycja zamyka sie i odwraca dopiero na przeciwnym sygnale, zgodnie z profilem WaveTrend
- wspiera backtest, walk-forward optimization oraz live/paper runner
- zachowuje dashboard z wizualizacja ceny, markerow transakcji oraz panelu `WT1/WT2`
- pozwala uruchomic zwykly backtest z recznie wybranymi parametrami strategii bez WFO

## Parametry WFO

W aktualnej wersji WFO testowane sa:

- `wt_channel_len`
- `wt_avg_len`
- `wt_signal_len`
- `wt_min_signal_level`
- `wt_long_entry_window_bars` / `wt_short_entry_window_bars`
- `wt_long_require_ema20_reclaim` / `wt_short_require_ema20_reject`
- `wt_ema_filter_len`
- `wt_long_entry_max_above_zero`
- `wt_short_entry_min_below_zero`

Domyslna siatka WFO:

- `channel_len`: `8, 10, 12`
- `avg_len`: `14, 21, 28`
- `signal_len`: `3, 4`
- `min_signal_level`: `0, 10`
- `re-entry window`: `1, 2`
- `EMA filter`: `off, on`
- `EMA length`: domyslnie `10, 20`; w dashboardzie mozna zaznaczyc tez `8, 15`
- `long zone max`: domyslnie `-10, -20`; w dashboardzie mozna zaznaczyc tez `-30, -40`
- `short zone min`: domyslnie `10, 20`; w dashboardzie mozna zaznaczyc tez `30, 40`

Dashboard pokazuje dodatkowe wskazniki jakosci strategii:

- `Return / Drawdown`
- `Sharpe Ratio`
- `Sortino Ratio`
- `Risk/Reward Ratio`
- `Expectancy`
- `Consistency / Stability`

Dashboard ma teraz dwa osobne tryby:

- `Backtest manualny`, gdzie ustawiasz te parametry recznie i liczysz zwykly backtest
- `WFO`, gdzie bot szuka najlepszych ustawien na oknie optymalizacyjnym i stosuje je na kolejnym oknie live

## Najwazniejsze pliki

- [bee7_main.py](bee7_main.py)
- [bee7_dashboard.py](bee7_dashboard.py)
- [bee7_strategy.py](bee7_strategy.py)
- [bee7_wfo.py](bee7_wfo.py)
- [bee7_live_runner.py](bee7_live_runner.py)

## Uruchomienie

```bash
python bee7_dashboard.py --host 0.0.0.0 --port 8064
```

```bash
python bee7_main.py --mode backtest
```

```bash
python bee7_main.py --mode wfo
```
