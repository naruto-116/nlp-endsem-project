"""
Quick Test Script for Both UIs
Run this to test Streamlit and Gradio interfaces
"""
import subprocess
import sys
from pathlib import Path

def test_streamlit():
    """Test Streamlit interface."""
    print("\n" + "="*80)
    print("TESTING STREAMLIT INTERFACE")
    print("="*80)
    print("\n🚀 Starting Streamlit on http://localhost:8501")
    print("📍 Press Ctrl+C to stop and continue to Gradio test\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n✅ Streamlit test stopped.")

def test_gradio():
    """Test Gradio interface."""
    print("\n" + "="*80)
    print("TESTING GRADIO INTERFACE")
    print("="*80)
    print("\n🚀 Starting Gradio on http://localhost:7860")
    print("📍 Press Ctrl+C to stop\n")
    
    try:
        subprocess.run([sys.executable, "app_gradio.py"])
    except KeyboardInterrupt:
        print("\n✅ Gradio test stopped.")

def main():
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                    KG-CiteRAG UI TESTING SCRIPT                            ║
╚════════════════════════════════════════════════════════════════════════════╝

This script will test both interfaces:
1. Streamlit UI (http://localhost:8501)
2. Gradio UI (http://localhost:7860)

Choose an option:
[1] Test Streamlit only
[2] Test Gradio only
[3] Test both (Streamlit first, then Gradio)
[4] Exit
    """)
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        test_streamlit()
    elif choice == "2":
        test_gradio()
    elif choice == "3":
        print("\n⚡ Testing Streamlit first...")
        test_streamlit()
        print("\n⚡ Now testing Gradio...")
        test_gradio()
    elif choice == "4":
        print("\n👋 Exiting. Have a great day!")
        sys.exit(0)
    else:
        print("\n❌ Invalid choice. Please run the script again.")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("✅ TESTING COMPLETE!")
    print("="*80)
    print("""
📊 RESULTS:
- Streamlit: http://localhost:8501
- Gradio: http://localhost:7860

📝 To deploy:
- See DEPLOYMENT_GUIDE.md for detailed instructions
- Recommended: Gradio + HuggingFace Spaces (FREE, 2GB RAM)

🚀 Quick Deploy (Gradio with public URL):
1. Edit app_gradio.py line 600: change share=False to share=True
2. Run: python app_gradio.py
3. Copy the public URL (valid 72 hours)
    """)

if __name__ == "__main__":
    main()
