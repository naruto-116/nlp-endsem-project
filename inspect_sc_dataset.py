"""
Script to inspect the Indian Supreme Court Judgments dataset
"""
import pandas as pd
import os
from pathlib import Path

# Load metadata
csv_path = "Indian Supreme Court Judgments/judgments.csv"
df = pd.read_csv(csv_path)

print("="*70)
print("DATASET OVERVIEW")
print("="*70)
print(f"Total judgments in CSV: {len(df)}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")

print("\n" + "="*70)
print("SAMPLE RECORDS")
print("="*70)
print(df.head(3))

print("\n" + "="*70)
print("DATA STATISTICS")
print("="*70)
print(f"Date range: {df['judgment_dates'].min()} to {df['judgment_dates'].max()}")
print(f"\nLanguages:\n{df['language'].value_counts()}")
print(f"\nUnique diary numbers: {df['diary_no'].nunique()}")
print(f"Unique case numbers: {df['case_no'].nunique()}")
print(f"\nJudgment types:\n{df['Judgement_type'].value_counts()}")

print("\n" + "="*70)
print("PDF FILES")
print("="*70)
pdf_dir = Path("Indian Supreme Court Judgments/pdfs")
if pdf_dir.exists():
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"Total PDF files: {len(pdf_files)}")
    
    # Check file sizes
    valid_pdfs = [f for f in pdf_files if f.stat().st_size > 1000]  # > 1KB
    empty_pdfs = [f for f in pdf_files if f.stat().st_size <= 1000]
    
    print(f"Valid PDFs (>1KB): {len(valid_pdfs)}")
    print(f"Empty/Small PDFs: {len(empty_pdfs)}")
    
    if valid_pdfs:
        print(f"\nSample valid PDFs:")
        for pdf in valid_pdfs[:5]:
            print(f"  {pdf.name} ({pdf.stat().st_size / 1024:.1f} KB)")
else:
    print("PDF directory not found")

print("\n" + "="*70)
print("MISSING DATA ANALYSIS")
print("="*70)
print(f"Missing values per column:")
print(df.isnull().sum())
