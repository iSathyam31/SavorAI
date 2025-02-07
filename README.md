## SavorAI: AI-driven Food Selection For Every Plate
### Overview
SavorAI is an intelligent restaurant menu assistant that leverages AI to provide personalized food recommendations and enhance dining experiences. By analyzing uploaded menu PDFs, the application offers context-aware suggestions, dietary considerations, and interactive dining guidance.

### Features
* 📄 PDF Menu Upload: Easily upload restaurant menus
* 🧠 Intelligent Recommendations: AI-powered food suggestions
* 💬 Conversational Interface: Natural language interactions
* 🔍 Context-Aware Responses: Understands menu-related and casual queries
* 🌿 Dietary Preference Support: Considers dietary restrictions and preferen

### Project Structure
```
SavorAI 
│── Data/                # Folder containing restaurant menu documents  
│── venv/                # Virtual environment (not included in the repository)  
│── .gitignore           # Git ignore file  
│── app.py               # Main application file  
│── LICENSE              # License information  
│── README.md            # Project documentation  
│── requirements.txt     # Dependencies  
```

### Setup Instructions
#### Create a Virtual Environment
```
python -m venv venv
source venv/bin/activate    # On macOS/Linux
venv\Scripts\activate       # On Windows
```

#### Install Dependencies
```
pip install -r requirements.txt
```

#### Install and Run Ollama
Ollama is required to run the LLM locally.
* Download and install Ollama: Ollama Official Site
* Start Ollama:
```
ollama run
```

#### Run the Project
```
python app.py
```

### How It Works
* **Ingest Menus**: The application loads menu documents from the Data/ directory.
* **User Query**: The user provides food preferences.
* **RAG Process**: The model retrieves relevant menu items and generates suggestions.
* **Output**: A personalized list of food recommendations.

### License
This project is licensed under the GNU GENERAL PUBLIC LICENSE .
