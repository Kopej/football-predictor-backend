# Football Predictor Backend

A FastAPI-based football prediction backend for:

- EPL
- LaLiga
- Serie A
- Bundesliga

It serves ranked multi-market predictions for each match, including:

- Match Result (Home / Draw / Away)
- Double Chance
- BTTS
- Over 2.5 Goals
- Under 4.5 Goals

## Project Structure

```text
football-predictor-backend/
  data/
    raw/
    processed/
  models/
  notebooks/
  src/
    __init__.py
    config.py
    data_loader.py
    validate.py
    features.py
    train.py
    train_markets.py
    predict.py
    api.py
  requirements.txt
  README.md
```
