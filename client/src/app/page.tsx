import Link from "next/link";
import { ArrowRight, Database, Dna, GitBranch } from "lucide-react";
import * as motion from "framer-motion/client";

export default function Home() {
  return (
    <div className="relative min-h-[calc(100vh-80px)] overflow-hidden flex flex-col justify-center px-6 py-24 sm:py-32 lg:px-12">
      
      {/* Background Graphic Element */}
      <div className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-1/3 opacity-5 pointer-events-none">
        <Dna className="w-[800px] h-[800px] text-primary" strokeWidth={0.5} />
      </div>

      <div className="max-w-4xl relative z-10">
        
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          className="inline-flex items-center gap-3 border border-border px-4 py-2 mb-8 bg-background/50 backdrop-blur-sm"
        >
          <span className="w-2 h-2 bg-primary animate-pulse" />
          <span className="text-xs uppercase tracking-[0.2em] font-bold text-muted-foreground">BCSE498J - Project II</span>
        </motion.div>

        <motion.h1 
          initial={{ opacity: 0, clipPath: "polygon(0 100%, 100% 100%, 100% 100%, 0% 100%)" }}
          animate={{ opacity: 1, clipPath: "polygon(0 0, 100% 0, 100% 100%, 0 100%)" }}
          transition={{ duration: 1.2, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
          className="font-display text-5xl sm:text-7xl lg:text-[5.5rem] font-bold tracking-tighter leading-[0.9] text-foreground mb-8"
        >
          ENHANCING SKIN LESION CLASSIFICATION<br />
          <span className="text-primary italic text-xl sm:text-3xl lg:text-4xl mt-6 block uppercase tracking-tight">USING DATA CENTRIC AI & SYNTHETIC IMAGES</span>
        </motion.h1>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.5 }}
          className="grid sm:grid-cols-2 gap-8 lg:gap-16 max-w-3xl"
        >
          <p className="text-muted-foreground text-lg sm:text-xl font-light leading-relaxed">
            A high-performance pipeline classifying 14 distinct skin lesions. Engineered to overcome severe dataset imbalances using modern geometric regularizations and focal loss strategies.
          </p>
          
          <div className="flex flex-col justify-center gap-6">
            <Link 
              href="/demo"
              className="group relative flex items-center justify-between border border-border bg-background p-4 overflow-hidden transition-colors hover:border-primary"
            >
              <div className="absolute inset-0 bg-primary translate-y-full transition-transform duration-300 ease-out group-hover:translate-y-0" />
              <span className="relative z-10 font-bold uppercase tracking-widest text-sm text-foreground group-hover:text-white transition-colors">Launch Interface</span>
              <ArrowRight className="relative z-10 w-5 h-5 text-foreground group-hover:text-white group-hover:translate-x-1 transition-all" />
            </Link>

            <div className="flex gap-4">
              <a 
                href="https://www.kaggle.com/datasets/nour12347653/skin-disease-detection-dataset-ham10000-isic" 
                target="_blank" 
                rel="noreferrer"
                className="flex-1 border border-border p-4 flex items-start gap-4 hover:bg-white hover:text-black transition-colors cursor-pointer group"
              >
                <Database className="w-6 h-6 shrink-0 group-hover:text-black" strokeWidth={1.5} />
                <div>
                  <div className="font-display font-bold text-lg leading-none mb-1">HAM10000</div>
                  <div className="text-[10px] uppercase tracking-wider text-muted-foreground group-hover:text-gray-600">Dataset Link</div>
                </div>
              </a>
              <a 
                href="https://github.com/kishorekrrish3/Enhanced-Skin-Lesion-Classification"
                target="_blank" 
                rel="noreferrer"
                className="flex-1 border border-border p-4 flex items-start gap-4 hover:bg-white hover:text-black transition-colors cursor-pointer group"
              >
                <GitBranch className="w-6 h-6 shrink-0 group-hover:text-black" strokeWidth={1.5} />
                <div>
                  <div className="font-display font-bold text-lg leading-none mb-1">GitHub</div>
                  <div className="text-[10px] uppercase tracking-wider text-muted-foreground group-hover:text-gray-600">Source Code</div>
                </div>
              </a>
            </div>
          </div>
        </motion.div>

      </div>
    </div>
  );
}
