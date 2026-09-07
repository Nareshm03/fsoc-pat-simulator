# PREPARE DATASET FOR COLAB TRAINING
# Run this script to zip the dataset and verify it's ready

Write-Host "`n$('='*80)" -ForegroundColor Cyan
Write-Host "  PREPARING DATASET FOR GOOGLE COLAB TRAINING" -ForegroundColor Cyan
Write-Host "$('='*80)`n" -ForegroundColor Cyan

# Step 1: Verify dataset exists and has correct counts
Write-Host "Step 1: Verifying dataset..." -ForegroundColor Yellow

$trainImages = (Get-ChildItem images\train\*.png).Count
$valImages = (Get-ChildItem images\val\*.png).Count
$testImages = (Get-ChildItem images\test\*.png).Count
$total = $trainImages + $valImages + $testImages

Write-Host "  Train: $trainImages images"
Write-Host "  Val:   $valImages images"
Write-Host "  Test:  $testImages images"
Write-Host "  Total: $total images"

if ($total -ne 10000) {
    Write-Host "`n  WARNING: Expected 10000 images, found $total" -ForegroundColor Red
    $continue = Read-Host "`nContinue anyway? (y/n)"
    if ($continue -ne 'y') {
        exit 1
    }
}

Write-Host "  Status: OK" -ForegroundColor Green

# Step 2: Verify dataset.yaml
Write-Host "`nStep 2: Verifying dataset.yaml..." -ForegroundColor Yellow

if (Test-Path "dataset.yaml") {
    Write-Host "  Found: dataset.yaml"
    $yaml = Get-Content "dataset.yaml" -Raw
    if ($yaml -match "path:\s*\.") {
        Write-Host "  Path: Relative (correct for Colab)" -ForegroundColor Green
    } else {
        Write-Host "  Path: Absolute (will work but not optimal)" -ForegroundColor Yellow
    }
    if ($yaml -match "nc:\s*1") {
        Write-Host "  Classes: 1 (beacon)" -ForegroundColor Green
    }
} else {
    Write-Host "  ERROR: dataset.yaml not found!" -ForegroundColor Red
    exit 1
}

# Step 3: Check for required files
Write-Host "`nStep 3: Checking required files..." -ForegroundColor Yellow

$requiredFiles = @(
    "dataset.yaml",
    "README.md",
    "train_yolo_colab.ipynb",
    "COLAB_TRAINING_GUIDE.md"
)

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  Found: $file" -ForegroundColor Green
    } else {
        Write-Host "  Missing: $file" -ForegroundColor Yellow
    }
}

# Step 4: Create ZIP file
Write-Host "`nStep 4: Creating dataset.zip..." -ForegroundColor Yellow

$zipPath = "..\dataset.zip"

if (Test-Path $zipPath) {
    Write-Host "  Removing old dataset.zip..."
    Remove-Item $zipPath -Force
}

Write-Host "  Compressing dataset (this may take a few minutes)..."

# Create ZIP with all contents
Compress-Archive -Path * -DestinationPath $zipPath -CompressionLevel Optimal -Force

if (Test-Path $zipPath) {
    $zipSize = [math]::Round((Get-Item $zipPath).Length / 1MB, 2)
    Write-Host "  Created: dataset.zip" -ForegroundColor Green
    Write-Host "  Size: $zipSize MB"
    
    if ($zipSize -gt 500) {
        Write-Host "  WARNING: ZIP file is very large (>500 MB)" -ForegroundColor Yellow
        Write-Host "  Consider using Google Drive for upload" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ERROR: Failed to create ZIP file!" -ForegroundColor Red
    exit 1
}

# Step 5: Summary
Write-Host "`n$('='*80)" -ForegroundColor Green
Write-Host "  DATASET READY FOR COLAB!" -ForegroundColor Green
Write-Host "$('='*80)`n" -ForegroundColor Green

Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Upload dataset.zip to Google Colab"
Write-Host "  2. Upload train_yolo_colab.ipynb"
Write-Host "  3. Enable GPU (Runtime > Change runtime type > T4)"
Write-Host "  4. Run cells in order"
Write-Host "  5. Monitor training (expect 3-5 hours)"
Write-Host "  6. Download best.pt and best.onnx"

Write-Host "`nFiles Ready:" -ForegroundColor Cyan
Write-Host "  ../dataset.zip           - Upload to Colab"
Write-Host "  train_yolo_colab.ipynb   - Upload to Colab"
Write-Host "  COLAB_TRAINING_GUIDE.md  - Read for detailed instructions"

Write-Host "`nExpected Results:" -ForegroundColor Cyan
Write-Host "  Training Time: 3-5 hours (T4 GPU)"
Write-Host "  mAP50: >0.95"
Write-Host "  mAP50-95: >0.80"
Write-Host "  Model Size: ~80-90 MB (PyTorch)"

Write-Host "`n$('='*80)`n" -ForegroundColor Green

# Optional: Open guide
$openGuide = Read-Host "Open COLAB_TRAINING_GUIDE.md? (y/n)"
if ($openGuide -eq 'y') {
    Start-Process "COLAB_TRAINING_GUIDE.md"
}
