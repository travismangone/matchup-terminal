# Matchup Terminal

Daily MLB pitch-type matchup tool. Pulls today's probable starters and lineups from the MLB Stats API, computes each pitcher's recent arsenal split by batter handedness, looks up each batter's wOBA and ISO against those specific pitch types (with Bayesian shrinkage toward league average), and ranks the slate by projected matchup quality.

## Features

- Pitcher arsenal split by batter handedness (lefty vs righty)
- Pitcher wOBA and ISO allowed vs LHB and vs RHB
- Batter wOBA and ISO by pitch type, last two seasons
- Bayesian shrinkage toward league average for small samples
- Projected wOBA edge and ISO edge per batter for today's matchup
- Sortable, heatmapped batter table; pitcher cards at the top

## Data Sources

- Baseball Savant (Statcast) via pybaseball
- MLB Stats API via MLB-StatsAPI

## Run Locally

```
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Deploy on Render

The repo includes a Procfile. Create a Web Service on Render pointing at this repo, set the build command to `pip install -r requirements.txt`, and Render will start the app via gunicorn on the platform-assigned port.
