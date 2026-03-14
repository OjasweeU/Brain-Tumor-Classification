# Production Deployment Guide

Deploying your full-stack application will involve two separate services:
1. **Frontend**: We will deploy the React application using **Vercel** (Free, easiest for frontend).
2. **Backend**: We will deploy the FastAPI python server using **Render** (Free, great for APIs).

Since you already have the repository on GitHub, both of these platforms can automatically deploy directly from your repo!

## Step 1: Deploy the Backend on Render
1. Go to [Render.com](https://render.com/) and sign up or log in using GitHub.
2. Click **New +** and select **Web Service**.
3. Select **"Build and deploy from a Git repository"** and connect your `Brain-Tumor-Classification` repository.
4. Fill in the following settings:
   - **Name**: `neuroai-backend` (or similar)
   - **Root Directory**: `backend` (This is very important!)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free
5. Open the **"Advanced"** dropdown, click **Add Environment Variable**, and add:
   - **Name**: `PYTHON_VERSION`
   - **Value**: `3.10.13`
6. Click **Create Web Service**.
6. Wait 3-5 minutes for the build to finish. Once it says "Live", copy the provided URL at the top left (it will look like `https://neuroai-backend-xxxx.onrender.com`).

*Note: Since the backend is running in Mock Mode, it will safely deploy within Render's free RAM limitations.*

## Step 2: Deploy the Frontend on Vercel
1. Go to [Vercel.com](https://vercel.com/) and sign up or log in using GitHub.
2. Click **Add New...** -> **Project**.
3. Import your `Brain-Tumor-Classification` GitHub repository.
4. In the configuration screen, click the **Edit** button next to **Root Directory** and select `frontend`.
5. Open the **"Environment Variables"** dropdown and add a new secret variable so the React app knows where the backend lives:
   - **Name**: `VITE_API_URL`
   - **Value**: `[PASTE_YOUR_RENDER_URL_HERE]` (e.g., `https://neuroai-backend-xxxx.onrender.com`)
6. Click **Deploy**.
7. Vercel will build the React site in about 1 minute. 

Congratulations! You now have a complete, production-ready live URL for both your backend and frontend that you can share on your resume. 🚀
