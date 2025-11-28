# KG-CiteRAG Quick Start Script
# This script helps you get started quickly

Write-Host "=" -NoNewline; for($i=0; $i -lt 59; $i++) { Write-Host "=" -NoNewline }; Write-Host ""
Write-Host "KG-CiteRAG Setup Assistant"
Write-Host "=" -NoNewline; for($i=0; $i -lt 59; $i++) { Write-Host "=" -NoNewline }; Write-Host ""
Write-Host ""

# Check Python
Write-Host "Checking Python installation..."
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Create .env if not exists
if (-not (Test-Path ".env")) {
    Write-Host "`nCreating .env file..."
    Copy-Item ".env.example" ".env"
    Write-Host "✓ Created .env file" -ForegroundColor Green
    Write-Host "⚠ Please edit .env and add your Groq API key" -ForegroundColor Yellow
} else {
    Write-Host "`n✓ .env file already exists" -ForegroundColor Green
}

# Create data directory
if (-not (Test-Path "data")) {
    New-Item -ItemType Directory -Path "data" | Out-Null
    Write-Host "✓ Created data directory" -ForegroundColor Green
}

# Ask about installation
Write-Host "`n" -NoNewline
$install = Read-Host "Install Python dependencies? (y/n)"
if ($install -eq "y") {
    Write-Host "`nInstalling dependencies..."
    pip install -r requirements.txt
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green
    } else {
        Write-Host "✗ Installation failed" -ForegroundColor Red
        exit 1
    }
}

# Check for ILDC data
Write-Host "`nChecking for ILDC dataset..."
if (-not (Test-Path "data\ILDC_single.jsonl")) {
    Write-Host "⚠ ILDC dataset not found" -ForegroundColor Yellow
    Write-Host "`nOptions:"
    Write-Host "1. Download ILDC_single.jsonl and place it in the data/ folder"
    Write-Host "2. Create a sample dataset for testing (recommended for first run)"
    Write-Host ""
    $choice = Read-Host "Choose option (1/2)"
    
    if ($choice -eq "2") {
        Write-Host "`nCreating sample dataset..."
        python -c 'from scripts.data_loader import create_sample_ildc_file; from pathlib import Path; import config; create_sample_ildc_file(config.ILDC_PATH, 100)'
        Write-Host "✓ Sample dataset created" -ForegroundColor Green
    } else {
        Write-Host "`nPlease download ILDC_single.jsonl and place it in data/ folder"
        Write-Host "Then run this script again"
        exit 0
    }
} else {
    Write-Host "✓ ILDC dataset found" -ForegroundColor Green
}

# Build Knowledge Graph
Write-Host "`n" -NoNewline
$buildGraph = Read-Host "Build Knowledge Graph? (y/n)"
if ($buildGraph -eq "y") {
    Write-Host "`nBuilding Knowledge Graph..."
    python scripts\build_knowledge_graph.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Knowledge Graph built successfully" -ForegroundColor Green
    } else {
        Write-Host "✗ Graph building failed" -ForegroundColor Red
        exit 1
    }
}

# Build Vector Index
Write-Host "`n" -NoNewline
$buildIndex = Read-Host "Build Vector Index? (y/n)"
if ($buildIndex -eq "y") {
    Write-Host "`nBuilding Vector Index..."
    python scripts\build_vector_index.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Vector Index built successfully" -ForegroundColor Green
    } else {
        Write-Host "✗ Index building failed" -ForegroundColor Red
        exit 1
    }
}

# Summary
Write-Host "`n" -NoNewline
Write-Host "=" -NoNewline; for($i=0; $i -lt 59; $i++) { Write-Host "=" -NoNewline }; Write-Host ""
Write-Host "Setup Complete!"
Write-Host "=" -NoNewline; for($i=0; $i -lt 59; $i++) { Write-Host "=" -NoNewline }; Write-Host ""
Write-Host "`nNext steps:"
Write-Host "1. Edit .env and add your Groq API key (if not done)"
Write-Host "2. Run the application: streamlit run app.py"
Write-Host ""
Write-Host "For detailed documentation, see:"
Write-Host "- SETUP.md (Setup guide)"
Write-Host "- USAGE.md (Usage examples)"
Write-Host "- README.md (Project overview)"
Write-Host ""

$run = Read-Host "Launch the application now? (y/n)"
if ($run -eq "y") {
    Write-Host ""
    Write-Host "Launching KG-CiteRAG..."
    streamlit run app.py
}
