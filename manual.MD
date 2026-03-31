# 📘 Project Manual: GitHub & Deployment

This guide will walk you through pushing your project to GitHub and deploying both the Frontend (Client) and Backend (Server) for free.

---

## 🚀 1. Pushing to GitHub

### Step 1: Initialize Git
Open your terminal in the root folder (`project-v2`) and run:
```powershell
git init
```

### Step 2: Add Files
We have already created a `.gitignore` to ensure large datasets (`data/`) and model weights are not pushed.
```powershell
git add .
```

### Step 3: Commit
```powershell
git commit -m "Initial commit: Revamped production structure"
```

### Step 4: Push to Remote
1. Create a **New Repository** on GitHub (do not initialize with README).
2. Copy the repository URL (e.g., `https://github.com/yourusername/your-repo.git`).
3. Run these commands:
```powershell
git remote add origin YOUR_REPOSITORY_URL
git branch -M main
git push -u origin main
```

---

## 🌐 2. Deployment Guide (Free Tier)

### 🎨 Frontend: Vercel (Recommended)
Vercel is the best platform for Next.js applications and offers a generous free tier.

1. Go to [Vercel.com](https://vercel.com) and sign up with GitHub.
2. Click **"Add New"** -> **"Project"**.
3. Import your GitHub repository.
4. **Project Settings**:
   - **Root Directory**: Select `client/`.
   - **Framework Preset**: Next.js.
   - **Environment Variables**:
     - `NEXT_PUBLIC_API_URL`: Your deployed Backend URL (e.g., `https://your-backend.onrender.com/api`).
5. Click **Deploy**.

### ⚙️ Backend: Render (Recommended)
Render is perfect for FastAPI/Python applications and has a reliable free tier.

1. Go to [Render.com](https://render.com) and sign up with GitHub.
2. Click **"New"** -> **"Web Service"**.
3. Connect your GitHub repository.
4. **Project Settings**:
   - **Name**: `skin-lesion-api` (or choice).
   - **Root Directory**: Select `server/`.
   - **Runtime**: `Python 3`.
   - **Build Command**: `pip install -r requirements.txt`.
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
5. Click **Create Web Service**.

---

## 🔑 3. Important Environment Variables

To allow the Frontend to communicate with the Backend in production, you must set these variables in their respective dashboards:

| Platform | Variable Name | Value |
| :--- | :--- | :--- |
| **Vercel** | `NEXT_PUBLIC_API_URL` | `https://your-backend-url.onrender.com/api` |
| **Render** | `CORS_ORIGINS` | `https://your-frontend-url.vercel.app` |

---

## ⚠️ Notes for Free Tiers
- **Spin-up Time**: Render's free tier "sleeps" after 15 mins of inactivity. The first request after a break might take 30-50 seconds to respond.
- **File Storage**: Free tiers generally have ephemeral file systems. Uploaded images or results generated at runtime will be lost if the server restarts. For permanent storage, consider Cloudinary or AWS S3.
