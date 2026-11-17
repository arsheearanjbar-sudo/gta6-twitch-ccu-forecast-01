# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/arsheearanjbar-sudo/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/arsheearanjbar-sudo/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day, 2026-11-19 (UTC), from the first hour's viewer count. One Python program (`main.py`, standard library only, deterministic) reads a case file and writes five quantiles per hour (5%, 25%, 50%, 75%, 95%) for hours 2 to 24.

How it works: the forecast anchors on the case's hour-1 print and learns the launch-day shape from the case's comparables — past big launches' hourly curves. For each hour it fits log(viewers / hour-1) against log(hour-1) across comparable rolling-midnight launches, so a small first hour (a US-heavy audience still asleep) forecasts a steep ramp and a big first hour forecasts a flatter day; the spread is the regression's residual standard deviation. No network, no randomness, no model calls.

Local validation (practice cases, scored with the contest's own pinball loss on the log scale): mean loss **0.1825** across the four practice launches (Wolverine 0.2408, Red Dead Redemption 2 0.1563, God of War Ragnarök 0.1420, Spider-Man 2 0.1909), against the house baseline's 0.3871. Leave-one-out over the comparables' own launch days: 0.2812 on rolling-midnight launches, against the baseline's 0.4228. Runs in well under a second per case; two runs write byte-identical output.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` reads every `*.in` case in the input directory and writes one `*.out` forecast per case to the output directory. With neither environment variable set it reads `/input` and writes `/output`, as the grader runs it.

---

<sub>Entered on [make.fun](https://make.fun/?ref=sh01ee1y&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
