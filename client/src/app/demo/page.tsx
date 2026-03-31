"use client";

import { useState, useRef } from "react";
import { predictImage } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Upload, X, Loader2, Maximize, ScanSearch, CheckCircle2 } from "lucide-react";
import * as motion from "framer-motion/client";

export default function Demo() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [modelName, setModelName] = useState("ImprovedResNet50");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };

  const handleClear = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  }

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await predictImage(file, modelName);
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to analyze image");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto py-16 px-4 max-w-[1400px]">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col md:flex-row gap-8 lg:gap-16 items-start"
      >
        
        {/* Left Column: Input */}
        <div className="w-full md:w-1/2 lg:w-5/12 space-y-8 sticky top-28">
          <div className="space-y-4">
            <h1 className="font-display text-5xl md:text-6xl font-black uppercase tracking-tighter leading-none">Diagnostic<br/>Console</h1>
            <p className="text-muted-foreground font-light text-lg border-l-2 border-primary pl-4 max-w-sm">Feed raw imagery to the pipeline. Monitor active layer tracking via Grad-CAM isolation.</p>
          </div>

          <div className="border border-border bg-background/50 backdrop-blur-sm p-6 space-y-8 relative group">
            <div className="absolute top-0 right-0 w-3 h-3 border-t-2 border-r-2 border-primary" />
            <div className="absolute bottom-0 left-0 w-3 h-3 border-b-2 border-l-2 border-primary" />
            
            <div className="space-y-4">
              <div className="flex justify-between items-center text-xs font-bold uppercase tracking-widest text-muted-foreground border-b border-border pb-2">
                <span>01. Protocol</span>
                <span>Architecture</span>
              </div>
              <Select value={modelName} onValueChange={setModelName}>
                <SelectTrigger className="w-full bg-transparent border-border rounded-none h-14 font-display text-lg uppercase focus:ring-primary focus:border-primary">
                  <SelectValue placeholder="Select Architecture" />
                </SelectTrigger>
                <SelectContent className="rounded-none border-border bg-background">
                  <SelectItem className="rounded-none cursor-pointer focus:bg-primary focus:text-primary-foreground font-display uppercase text-sm h-10" value="ImprovedResNet50">Phase 3: Optimized ResNet50</SelectItem>
                  <SelectItem className="rounded-none cursor-pointer focus:bg-primary focus:text-primary-foreground font-display uppercase text-sm h-10" value="BaselineResNet">Phase 2: Base ResNet50</SelectItem>
                  <SelectItem className="rounded-none cursor-pointer focus:bg-primary focus:text-primary-foreground font-display uppercase text-sm h-10" value="SimpleCNN">Phase 1: Baseline CNN</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-4">
              <div className="flex justify-between items-center text-xs font-bold uppercase tracking-widest text-muted-foreground border-b border-border pb-2">
                <span>02. Visual Data</span>
                <span>Upload</span>
              </div>
              
              {!previewUrl ? (
                <div 
                  className="border border-dashed border-muted-foreground/50 hover:border-primary bg-background p-12 flex flex-col items-center justify-center text-center cursor-pointer transition-colors group/upload"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <ScanSearch className="h-10 w-10 text-muted-foreground group-hover/upload:text-primary mb-6 transition-colors" strokeWidth={1} />
                  <p className="font-display font-medium text-lg uppercase tracking-wider mb-2 group-hover/upload:text-primary transition-colors">Select Scan Target</p>
                  <p className="text-muted-foreground text-sm font-light">Supported formats: JPG, PNG</p>
                  <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept="image/*" />
                </div>
              ) : (
                <div className="relative border border-border bg-background p-4 group/preview">
                  <div className="aspect-square w-full relative overflow-hidden bg-black flex items-center justify-center">
                    <img src={previewUrl} alt="Target" className="w-full h-full object-cover grayscale contrast-125 opacity-80" />
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                      <div className="w-16 h-16 border border-primary rounded-full animate-ping opacity-20" />
                      <div className="absolute w-2 h-2 bg-primary animate-pulse" />
                    </div>
                  </div>
                  <div className="mt-4 flex justify-between items-center text-xs font-bold uppercase tracking-widest">
                    <span className="text-primary flex items-center gap-2"><CheckCircle2 className="w-4 h-4"/> Target Locked</span>
                    <button onClick={handleClear} className="text-muted-foreground hover:text-destructive flex items-center gap-1"><X className="w-3 h-3" /> Clear</button>
                  </div>
                </div>
              )}
            </div>

            <Button 
              onClick={handlePredict} 
              disabled={!file || loading}
              className="w-full h-16 rounded-none bg-primary hover:bg-white hover:text-black text-primary-foreground font-display font-bold text-xl tracking-widest uppercase transition-all"
            >
              {loading ? (
                <><Loader2 className="mr-3 h-6 w-6 animate-spin" /> Processing Source...</>
              ) : (
                <>Run Diagnostic Inference</>
              )}
            </Button>
            
            {error && <div className="border border-destructive bg-destructive/10 text-destructive p-4 text-sm font-bold tracking-widest uppercase">{error}</div>}
          </div>
        </div>

        {/* Right Column: Output */}
        <div className="w-full md:w-1/2 lg:w-7/12 min-h-[600px] border border-border bg-background/50 p-6 lg:p-12 relative flex flex-col justify-center">
          <div className="absolute top-0 right-0 w-32 h-32 bg-primary/5 blur-[100px] pointer-events-none" />
          
          <div className="text-xs font-bold uppercase tracking-widest text-muted-foreground border-b border-border pb-4 mb-12 flex items-center justify-between">
            <span>Results</span>
            <span className="flex items-center gap-2 font-mono"><span className="w-2 h-2 rounded-full bg-primary animate-pulse"/> LIVE MONITOR</span>
          </div>

          {!result && !loading && (
            <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground space-y-6 opacity-30">
              <Maximize className="w-24 h-24 stroke-1" />
              <p className="font-display text-2xl uppercase tracking-widest font-light">Awaiting Data Feed</p>
            </div>
          )}

          {loading && (
            <div className="flex-1 flex items-center justify-center">
               <div className="text-center space-y-6">
                 <Loader2 className="w-16 h-16 text-primary animate-spin mx-auto stroke-1" />
                 <p className="font-display font-light uppercase tracking-[0.3em] text-muted-foreground animate-pulse">Computing Matrix...</p>
               </div>
            </div>
          )}

          {result && (
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
              className="flex-1 space-y-12"
            >
              <div className="grid lg:grid-cols-2 gap-8 items-stretch">
                {/* Heatmap Area */}
                <div className="flex-1">
                  <div className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-4">Grad-CAM Extraction [layer4]</div>
                  {result.heatmap ? (
                     <div className="aspect-square relative border border-primary p-2 group bg-obsidian">
                       <img src={result.heatmap} alt="Grad-CAM" className="w-full h-full object-cover mix-blend-screen filter contrast-125 saturate-150" />
                       <div className="absolute bottom-4 right-4 bg-primary text-primary-foreground px-3 py-1 font-mono text-xs font-bold uppercase">
                         Visual Proof
                       </div>
                     </div>
                  ) : (
                    <div className="aspect-square border border-border flex items-center justify-center text-center p-8 text-muted-foreground">
                      No tracing available for this architectural version. All data remains purely mathematical.
                    </div>
                  )}
                </div>

                {/* Data Dump */}
                <div className="flex flex-col justify-center space-y-8">
                  <div>
                    <div className="text-[10px] font-bold uppercase tracking-widest text-primary mb-2">Primary Diagnosis Vector</div>
                    <div className="font-display text-3xl xl:text-4xl uppercase font-black leading-tight break-words border-l-4 border-primary pl-4">{result.prediction}</div>
                  </div>
                  
                  <div>
                    <div className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-3 border-b border-border pb-2">Confidence Scores</div>
                    <div className="space-y-4 max-h-[250px] pr-4 overflow-y-auto custom-scrollbar">
                      {result.all_probabilities.map((item: any, idx: number) => (
                        <div key={idx} className="space-y-2 opacity-80 hover:opacity-100 transition-opacity">
                          <div className="flex justify-between items-end">
                            <span className={`text-[10px] font-bold uppercase tracking-widest ${idx === 0 ? "text-primary" : "text-muted-foreground"}`}>{item.class}</span>
                            <span className={`font-mono text-sm leading-none ${idx === 0 ? 'text-foreground' : 'text-muted-foreground'}`}>{(item.probability * 100).toFixed(1)}%</span>
                          </div>
                          <div className="h-1 bg-border w-full relative overflow-hidden">
                            <motion.div 
                               initial={{ width: 0 }}
                               animate={{ width: `${item.probability * 100}%` }}
                               transition={{ duration: 1, delay: idx * 0.1 }}
                               className={`absolute top-0 left-0 h-full ${idx === 0 ? 'bg-primary' : 'bg-muted-foreground'}`} 
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

        </div>
      </motion.div>
    </div>
  );
}
