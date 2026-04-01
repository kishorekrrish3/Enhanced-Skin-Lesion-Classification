/**
 * API utility for communicating with the FastAPI backend.
 */

// We assume the backend is running on localhost:8000 during development
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

export async function fetchMetrics() {
  try {
    const res = await fetch(`${API_BASE_URL}/metrics`, { cache: 'no-store' });
    if (!res.ok) throw new Error("Failed to fetch metrics");
    return await res.json();
  } catch (error) {
    console.error("Error fetching metrics:", error);
    return null;
  }
}

export async function predictImage(file: File, modelName: string) {
  try {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("model_name", modelName);

    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errorData = await res.json();
      throw new Error(errorData.detail || "Prediction failed");
    }

    return await res.json();
  } catch (error) {
    console.error("Error during prediction:", error);
    throw error;
  }
}
