# DEBUG
DEBUG: Bug Detection and Fix Suggestion Tool
DEBUG is an AI-powered solution designed to analyze source code, detect potential bugs, and provide intelligent suggestions for fixing them. This project leverages state-of-the-art machine learning models and natural language processing techniques to help developers improve code quality and streamline debugging.

Key Features
 -Bug Detection:
    -Uses a trained deep learning model (based on CodeBERT) to classify whether a code snippet is "buggy" or "bug-free."
    -Handles a variety of errors, including syntax issues, logical bugs, and runtime errors.
    
 -Fix Suggestion:
    -Provides intelligent recommendations for fixing detected bugs using a pre-trained generative model like CodeGPT.
    -Generates fixes for both simple and complex code issues
    
 -Graphical User Interface (GUI):
    -A user-friendly GUI built using Tkinter for easy code input and output visualization.
    -Enables local execution without relying on terminal commands.

    
Installation

1.Clone the Repository:
   git clone https://github.com/Rohan708/DEBUG.git
   
   cd DEBUG
   
2. Install Dependencies: Use pip to install the required Python libraries:
   
   pip install -r requirements.txt
   
3.Download Pre-Trained Models: Ensure the required models (e.g., CodeBERT, CodeGPT) are downloaded and accessible in the correct directory.

4.Run the Application:
  -cli:  python cli.py
  -APP:  python -m uvicorn app:app --reload
License
This project is released under a permissive open-source license (e.g., MIT), ensuring developers can use and extend the tool freely.

Acknowledgments
DEBUG was built using cutting-edge machine learning frameworks and pre-trained models like CodeBERT and CodeGPT. Special thanks to open-source contributors who make such advancements possible.
--------------------------Important Note---------------------------
The current version of the model is not fully trained due to limited computational resources. As a result, it may produce inconsistent or incorrect outputs when analyzing code snippets for bugs. This project is still under development, and we aim to refine the model and improve its accuracy in future updates
