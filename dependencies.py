# Save this as a .py file and run it to install the required libraries
import os

# List of required packages
packages = [
    "transformers",
    "torch",
    "scikit-learn",
    "pandas",
    "fastapi",  # Added FastAPI
    "pydantic"  # Added Pydantic
]

# Install each package
for package in packages:
    try:
        print(f"Installing {package}...")
        os.system(f"pip install {package}")
        print(f"{package} installed successfully.")
    except Exception as e:
        print(f"Failed to install {package}. Error: {e}")

print("All required packages have been installed!")