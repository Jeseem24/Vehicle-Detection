# Smart Campus Vehicle Management System

## Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Run the System
```bash
python main.py --demo
```

### 3. Open Dashboard
Open `frontend/index.html` in your browser.

---

## Hardware Optimization
- Optimized for AMD RX 6500M GPU using DirectML
- Fallback to CPU if GPU not available

## Project Structure
```
VD2/
├── backend/          # Python detection & tracking
├── frontend/         # Web dashboard
├── logs/             # CSV vehicle logs
└── demo/             # Demo videos
```
