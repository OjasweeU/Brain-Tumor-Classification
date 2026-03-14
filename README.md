# Brain Tumor Classification Full-Stack App

This project provides a complete frontend and backend solution for the Brain Tumor Classification model, built with React (Vite) and FastAPI.

## Folder Structure
- `backend/` - FastAPI server handling inference and Grad-CAM generation.
- `frontend/` - React application built with Vite for an aesthetic, glassmorphism UI.

## Getting Started

### 1. Backend Setup
The backend requires Python and some ML libraries.
1. Open a terminal and navigate to the project directory:
   ```bash
   cd "Brain Tumor Classification/backend"
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 2. Model Training (Optional, but Recommended)
Currently, if `model.h5` is not found, the backend will run in a **MOCK MODE** (returning dummy classifications to test the UI). To use the real Deep Learning model:
1. Run the training script:
   ```bash
   python train.py
   ```
   *Note: This will download the dataset if it's not present, process it, and train the EfficientNetB1 model, saving it as `model.h5`.*

### 3. Running the Backend Server
Once dependencies are installed:
```bash
python -m uvicorn main:app --reload
```
The FastAPI backend will start on `http://localhost:8000`.

### 4. Running the Frontend Server
1. Open a **new** terminal window and navigate to the frontend directory:
   ```bash
   cd "Brain Tumor Classification/frontend"
   ```
2. Install Node dependencies (if not already done):
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
4. Open your browser and go to `http://localhost:5173`. 

Upload an MRI scan to see the classification and the Grad-CAM activation visualization!
