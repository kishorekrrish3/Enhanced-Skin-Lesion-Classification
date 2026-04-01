/**
 * API utility for communicating with the FastAPI backend.
 */

interface PerformanceMetrics {
  accuracy: number;
  macro_recall: number;
  weighted_f1: number;
}

interface AllMetrics {
  SimpleCNN: PerformanceMetrics;
  BaselineResNet: PerformanceMetrics;
  ImprovedResNet50: PerformanceMetrics;
}

interface ProbItem {
  class: string;
  probability: number;
}

interface PredictionResult {
  prediction: string;
  heatmap?: string;
  all_probabilities: ProbItem[];
}

// We assume the backend is running on localhost:8000 during development
let API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

// Fix for production deployments where the trailing /api might be missing in the env var
if (API_BASE_URL && API_BASE_URL.startsWith('http')) {
  API_BASE_URL = API_BASE_URL.replace(/\/$/, ''); // strip trailing slash
  if (!API_BASE_URL.endsWith('/api')) {
    API_BASE_URL = API_BASE_URL + '/api';
  }
}

export async function fetchMetrics(): Promise<AllMetrics | null> {
  try {
    const res = await fetch(`${API_BASE_URL}/metrics`, { cache: 'no-store' });
    if (!res.ok) throw new Error("Failed to fetch metrics");
    return await res.json();
  } catch (error) {
    console.error("Error fetching metrics:", error);
    return null;
  }
}

export async function predictImage(file: File, modelName: string): Promise<PredictionResult> {
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
