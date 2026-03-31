"use client";

import { useEffect, useState } from "react";
import { fetchMetrics } from "@/lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Loader2, AlertCircle, ActivitySquare, Network, Layers } from "lucide-react";
import * as motion from "framer-motion/client";

const STATIC_BASE_URL = process.env.NEXT_PUBLIC_API_URL?.replace("/api", "/static") || "http://localhost:8000/static";

export default function Dashboard() {
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchMetrics().then((data) => {
      setMetrics(data);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-10 w-10 animate-spin text-primary" />
          <span className="font-display tracking-[0.2em] text-sm uppercase text-muted-foreground animate-pulse">Establishing Neural Link...</span>
        </div>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] text-center space-y-6">
        <AlertCircle className="h-16 w-16 text-destructive" strokeWidth={1} />
        <h2 className="text-4xl font-display font-bold uppercase tracking-tighter">Connection Severed</h2>
        <p className="text-muted-foreground max-w-md font-light">Ensure the clinical backend is active and the model pipeline has published its metrics JSON to the endpoint.</p>
      </div>
    );
  }

  const renderModelCard = (name: string, data: any, icon: any, isHighlight = false) => {
    if (!data) return null;
    const Icon = icon;
    return (
      <Card className={`relative overflow-hidden bg-background border-border group ${isHighlight ? 'border-primary lg:col-span-2 shadow-[0_0_30px_rgba(234,88,12,0.1)]' : ''}`}>
        {isHighlight && <div className="absolute top-0 right-0 w-32 h-32 bg-primary/10 blur-3xl rounded-full" />}
        <CardHeader className="pb-4 relative z-10">
          <div className="flex items-center gap-3">
            <Icon className={`w-6 h-6 ${isHighlight ? 'text-primary' : 'text-muted-foreground'}`} strokeWidth={1.5} />
            <CardTitle className={`font-display text-xl uppercase tracking-widest ${isHighlight ? 'text-foreground' : 'text-muted-foreground'}`}>{name}</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="relative z-10">
          <div className={`grid ${isHighlight ? 'sm:grid-cols-3' : 'grid-cols-1'} gap-6`}>
            <div className="flex flex-col gap-1 border-t border-border pt-4">
              <span className="text-muted-foreground text-xs font-bold uppercase tracking-widest">Global Accuracy</span>
              <span className="font-display text-3xl font-light">{(data.accuracy * 100).toFixed(1)}<span className="text-muted-foreground text-sm">%</span></span>
            </div>
            <div className="flex flex-col gap-1 border-t border-border pt-4">
              <span className="text-muted-foreground text-xs font-bold uppercase tracking-widest">Macro Recall</span>
              <span className={`font-display text-3xl font-bold ${isHighlight ? 'text-primary' : ''}`}>
                {(data.macro_recall * 100).toFixed(1)}<span className="text-muted-foreground text-sm font-light">%</span>
              </span>
            </div>
            <div className="flex flex-col gap-1 border-t border-border pt-4">
              <span className="text-muted-foreground text-xs font-bold uppercase tracking-widest">Weighted F1</span>
              <span className="font-display text-3xl font-light">{(data.weighted_f1 * 100).toFixed(1)}<span className="text-muted-foreground text-sm">%</span></span>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="container mx-auto py-16 px-4 md:px-8 space-y-16 max-w-[1400px]"
    >
      
      <div className="space-y-4 border-l-4 border-primary pl-6">
        <h1 className="font-display text-5xl md:text-6xl font-black tracking-tighter uppercase leading-none">Diagnostic<br/>Metrics</h1>
        <p className="text-muted-foreground max-w-2xl text-lg font-light">Performance tracing across architecture iterations. Emphasizing recall stabilization on minority classes.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {renderModelCard("Baseline CNN", metrics.SimpleCNN, Layers)}
        {renderModelCard("ResNet50 Base", metrics.BaselineResNet, Network)}
        {renderModelCard("ResNet50 Optimized", metrics.ImprovedResNet50, ActivitySquare, true)}
      </div>

      <div className="border-t border-border pt-16">
        <Tabs defaultValue="improved" className="w-full">
          <TabsList className="bg-transparent border-b border-border w-full justify-start h-auto p-0 mb-8 rounded-none">
            <TabsTrigger value="improved" className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:text-primary px-8 py-4 font-display uppercase tracking-widest text-sm">ResNet50 Analytics</TabsTrigger>
            <TabsTrigger value="baseline" className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent px-8 py-4 font-display uppercase tracking-widest text-sm">Baseline Diagnostics</TabsTrigger>
          </TabsList>
          
          <TabsContent value="improved" className="space-y-8 mt-0 data-[state=active]:animate-in data-[state=active]:fade-in data-[state=active]:slide-in-from-bottom-4 duration-500">
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
              <div className="border border-border p-6 bg-background/50 hover:bg-background transition-colors relative group">
                <div className="absolute top-0 left-0 w-2 h-2 border-t-2 border-l-2 border-primary" />
                <div className="absolute bottom-0 right-0 w-2 h-2 border-b-2 border-r-2 border-primary" />
                <div className="flex justify-between items-center mb-8 border-b border-border pb-4">
                  <h3 className="font-display text-2xl uppercase tracking-tighter">Confusion Matrix</h3>
                  <Badge variant="outline" className="border-primary text-primary rounded-none font-bold tracking-widest">TEST SPLIT</Badge>
                </div>
                <div className="aspect-square w-full relative mix-blend-screen">
                  <img 
                    src={`${STATIC_BASE_URL}/plots/ImprovedResNet50_test_cm.png`} 
                    alt="Confusion Matrix" 
                    className="absolute inset-0 w-full h-full object-contain filter contrast-125 saturate-0 group-hover:saturate-100 transition-all duration-700"
                    onError={(e) => { e.currentTarget.src = `${STATIC_BASE_URL}/plots/ImprovedResNet50_val_cm.png`; }}
                  />
                </div>
              </div>

              <div className="border border-border p-6 bg-background/50 hover:bg-background transition-colors relative group">
                <div className="absolute top-0 left-0 w-2 h-2 border-t-2 border-l-2 border-primary" />
                <div className="absolute bottom-0 right-0 w-2 h-2 border-b-2 border-r-2 border-primary" />
                <div className="flex justify-between items-center mb-8 border-b border-border pb-4">
                  <h3 className="font-display text-2xl uppercase tracking-tighter">Class Distribution Recall</h3>
                </div>
                <div className="aspect-square w-full relative mix-blend-screen">
                  <img 
                    src={`${STATIC_BASE_URL}/plots/ImprovedResNet50_test_recall.png`} 
                    alt="Per-Class Recall" 
                    className="absolute inset-0 w-full h-full object-contain filter contrast-125 saturate-0 group-hover:saturate-100 transition-all duration-700"
                    onError={(e) => { e.currentTarget.src = `${STATIC_BASE_URL}/plots/ImprovedResNet50_val_recall.png`; }}
                  />
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="baseline" className="mt-0 data-[state=active]:animate-in data-[state=active]:fade-in duration-500">
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
              <div className="border border-border p-6 bg-background max-w-3xl">
                <h3 className="font-display text-2xl uppercase tracking-tighter mb-8 border-b border-border pb-4 text-muted-foreground">Baseline Performance Degradation</h3>
                <div className="aspect-square w-full relative mix-blend-screen opacity-50">
                  <img 
                    src={`${STATIC_BASE_URL}/plots/SimpleCNN_test_cm.png`} 
                    alt="Baseline CM" 
                    className="absolute inset-0 w-full h-full object-contain grayscale"
                    onError={(e) => { e.currentTarget.src = `${STATIC_BASE_URL}/plots/SimpleCNN_val_cm.png`; }}
                  />
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </motion.div>
  );
}
