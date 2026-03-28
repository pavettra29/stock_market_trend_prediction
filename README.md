cat > README.md << 'EOF'
# Stock Market Trend Prediction

A  project that predicts stock market trends using an LSTM deep learning model.

## Tech Stack
- **Frontend**: React.js
- **Backend**: Flask (REST API)
- **ML Model**: LSTM (PyTorch)

## Project Structure
- `src/` — Python ML training code
- `api/` — Flask API server
- `frontend/` — React UI
- `models/` — Trained model weights (not included, see below)
- `data/` — Dataset (not included)

## How to Run
### Backend
```bash
cd api
pip install -r requirements.txt
python app.py
```
### Frontend
```bash
cd frontend
npm install
npm start
```

## Model
The LSTM model is trained on historical stock data. Pre-trained weights are not included in this repo due to file size. Run `src/train.py` to train from scratch.
EOF

git add README.md
git commit -m "Add README"
git push
