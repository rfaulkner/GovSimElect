#! /bin/bash 
 
 
echo "🚀 Starting setup process..." 
 
 
# Create and activate virtual environment 
echo "🌍 Creating Python virtual environment..." 
module load python/3.11.5 cuda/12.2 gcc arrow/21.0.0 rust 
virtualenv --no-download .venv --prompt GovSimElect 
echo "  → Activating virtual environment..." 
source .venv/bin/activate 
echo "✅ Virtual environment created and activated" 
 
 
# Install dependencies 
echo "📚 Installing dependencies..." 
echo "  → Installing project requirements..." 
echo "  → This may take a few minutes..." 
pip install -r requirements_DRAC.txt --find-links https://pypi.org/simple/ --prefer-binary 
echo "✅ All dependencies installed" 
 
 
# Optional login to Hugging Face 
echo "🔑 Logging into Hugging Face..." 
read -p "Do you want to log into Hugging Face to download gated models? (y/n): " 
if [[ $REPLY =~ ^[Yy]$ ]]; then 
   huggingface-cli login 
else 
   echo "Skipping Hugging Face login." 
fi 
echo "✅ Successfully logged into Hugging Face" 
 
 
echo "🎉 Setup completed successfully!" 
echo "💡 Virtual environment is now created. Activate it with: source .venv/bin/activate"
