# MedLens AI - Automated Setup Script
Write-Host "🚀 Starting MedLens AI Professional Setup..." -ForegroundColor Cyan

# 1. Backend Setup
Write-Host "`n📦 Setting up Backend..." -ForegroundColor Yellow
cd backend
if (-not (Test-Path venv)) {
    python -m venv venv
    Write-Host "✅ Virtual environment created."
}
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "⚠️ Created .env from example. Please add your GOOGLE_API_KEY." -ForegroundColor Magenta
}
Write-Host "✅ Backend dependencies installed."

# 2. Train Classifier
Write-Host "`n🧠 Training Intent Classifier..." -ForegroundColor Yellow
python src/train_classifier.py
Write-Host "✅ Classifier trained and saved."

# 3. Frontend Setup
cd ..
Write-Host "`n🎨 Setting up Frontend..." -ForegroundColor Yellow
cd frontend
npm install
Write-Host "✅ Frontend dependencies installed."

cd ..
Write-Host "`n✨ Setup Complete! ✨" -ForegroundColor Green
Write-Host "To start the application:"
Write-Host "1. Backend: cd backend; uvicorn src.api:app --reload"
Write-Host "2. Frontend: cd frontend; npm run dev"
