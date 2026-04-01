# 🚀 Deployment and GitHub Setup Manual

This manual provides clear, step-by-step instructions on pushing your capstone project to GitHub (ensuring large files like weights and datasets are excluded) and deploying the entire website—Frontend (Next.js) and Backend (FastAPI)—completely for free.

---

## 📂 Phase 1: Pushing Code to GitHub Safely

We have created an updated `.gitignore` file that safely ignores large folders like `weights/`, `data/`, `results/`, and `node_modules/`. This prevents GitHub size limit errors and makes your repository look highly professional.

Open your terminal in the root folder (`D:\ACADEMICS\CAPSTONE PROJECT\project-v2`) and follow these commands:

### 1. Initialize the Git Repository
```bash
git init
```

### 2. Stage All Allowed Files
```bash
git add .
```
> **Note**: Verify that folders like `weights` and `data` are NOT staged. Your `.gitignore` prevents them automatically.

### 3. Create First Commit
```bash
git commit -m "feat: initial commit for skin lesion classification capstone project"
```

### 4. Link to Your GitHub Repository
Go to [GitHub](https://github.com/new) and create a new repository without adding a README or .gitignore (leave it blank). Then:
```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
git push -u origin main
```

*(You now have a clean, lightweight GitHub repository!)*

---

## 🌐 Phase 2: Deploying the Backend (FastAPI) for Free

For the backend, we will use **Render** (Render.com) because it natively supports Python and offers a fantastic free tier.

### 1. Prepare for Deployment
Your backend lives in the `server/` directory and requires dependencies. 
Make sure your `requirements.txt` is updated in the root directory. If not, generate one that only covers production dependencies. Next, create a `render.yaml` file (Infrastructure as Code) if you want automated setup, or just use the dashboard.

### 2. Deploy via Render UI
1. Create a free account at [Render](https://render.com/).
2. Click **New +** > **Web Service**.
3. Connect your GitHub account and select your repository.
4. Fill in the deployment details:
   - **Name**: `skin-lesion-api` (or any name you prefer)
   - **Environment**: `Python`
   - **Region**: Pick the one closest to your location.
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn server.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free ($0/month)`
5. Click **Create Web Service**.

> ⚠️ **Important Step for Weights**: 
> Because your `weights/` folder isn't pushed to GitHub (it's too large), the deployed server won't have the `.pth` files to perform inference!
> 
> **Solution**: To keep hosting free, you should zip your `weights/` folder and upload it to a free file storage bucket (like Google Drive, AWS S3 free tier, or HuggingFace Models). Then, update your backend's initialization logic to automatically download the `.pth` files on startup if they don't exist in the local directory. (Let me know if you need code for this).

Once deployed successfully, Render will give you a public URL like: `https://skin-lesion-api.onrender.com`.

---

## 🖥 Phase 3: Deploying the Frontend (Next.js) for Free

For the frontend, **Vercel** is the gold standard for Next.js and provides an incredibly generous free tier.

### 1. Update Production APIs
Before pushing your Next.js app to Vercel, it needs to point to your new backend. Right now, your frontend probably references `http://localhost:8000`. You need to point it to the Render URL.

Go to `client/.env.local` (or create it) and add:
```env
NEXT_PUBLIC_API_URL=https://skin-lesion-api.onrender.com
```

Change your API fetch calls in the Next.js app to use `process.env.NEXT_PUBLIC_API_URL` instead of localhost. **Commit and push these changes to GitHub.**

### 2. Deploy via Vercel UI
1. Go to [Vercel](https://vercel.com/) and create a free account (sign up using GitHub).
2. Click **Add New** > **Project**.
3. Select your GitHub repository containing the project.
4. In the **Configure Project** step:
   - **Framework Preset**: `Next.js`
   - **Root Directory**: Click "Edit" and change it to `client/`
5. In **Environment Variables**, add:
   - Name: `NEXT_PUBLIC_API_URL`
   - Value: `https://skin-lesion-api.onrender.com`
6. Click **Deploy**.

Vercel will build your Next.js application and provide a free domain like `https://skin-lesion.vercel.app`.

---

## 🧹 Quick Recap of Project Reorganization
Your project structure is now highly professional:
- **`client/`**: Contains the Next.js Web Dashboard.
- **`server/`**: Contains the FastAPI logic and endpoints.
- **`core/`**: Contains the ML models, Dataset processing, and Pipeline utilities.
- **`data/` & `weights/`**: Ignored locally but securely stores the heavy files.
- **`docs/`**: Keep manuals or research findings here.
- **`.gitignore`**: Blocks unwanted cache / temp folders.

*If you need help with the auto-download code for the backend weights, let me know!*
