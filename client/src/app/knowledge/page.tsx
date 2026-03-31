import { ShieldAlert, GitBranch, Cpu, Stethoscope, Microscope, BrainCircuit, ExternalLink } from "lucide-react";
import * as motion from "framer-motion/client";

export default function KnowledgeBase() {
  return (
    <div className="container mx-auto py-16 px-4 md:px-8 max-w-[1400px]">
      
      <header className="mb-20 border-b-2 border-primary/20 pb-12">
        <div className="flex items-center gap-3 text-primary font-bold tracking-[0.2em] text-xs uppercase mb-4">
          <BrainCircuit className="w-5 h-5" />
          Neural Documentation
        </div>
        <h1 className="font-display text-5xl md:text-7xl font-black uppercase tracking-tighter leading-tight text-foreground">
          Project <span className="text-primary italic">Glossary</span>
        </h1>
        <p className="mt-6 text-lg font-light text-muted-foreground max-w-2xl leading-relaxed">
          A plain-English guide to the medical and technical layers of our classification pipeline. Designed for clinicians and researchers alike.
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        
        {/* MEDICAL 101 */}
        <motion.div 
          whileHover={{ y: -5 }}
          className="lg:col-span-2 border border-border bg-background p-8 flex flex-col justify-between group transition-all hover:border-primary/50"
        >
          <div>
            <Stethoscope className="w-10 h-10 text-primary mb-6" />
            <h2 className="font-display text-3xl font-bold uppercase tracking-tighter mb-4">Clinical 101</h2>
            <p className="text-muted-foreground leading-relaxed">
              Dermatology focuses on identifying patterns in skin cells. In this project, we classify <span className="text-foreground font-medium underline decoration-primary/30">14 distinct lesions</span> (spots), ranging from harmless birthmarks (Nevi) to critical malignancies like Melanoma.
            </p>
          </div>
          <div className="mt-8 pt-6 border-t border-border flex items-center justify-between text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
            <span>Domain: Medicine</span>
            <span className="text-primary">Essentials</span>
          </div>
        </motion.div>

        {/* MACHINE LEARNING 101 */}
        <motion.div 
          whileHover={{ y: -5 }}
          className="lg:col-span-2 border border-border bg-background p-8 flex flex-col justify-between group transition-all hover:border-primary/50"
        >
          <div>
            <Cpu className="w-10 h-10 text-primary mb-6" />
            <h2 className="font-display text-3xl font-bold uppercase tracking-tighter mb-4">Neural Networks</h2>
            <p className="text-muted-foreground leading-relaxed">
              Think of the AI as a digital brain that &quot;looks&quot; at thousands of lesion photos. It learns textures, borders, and color variants to calculate a <span className="text-foreground font-medium underline decoration-primary/30">Probability Score</span>—essentially telling us how sure it is about a diagnosis.
            </p>
          </div>
          <div className="mt-8 pt-6 border-t border-border flex items-center justify-between text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
            <span>Domain: AI/DL</span>
            <span className="text-primary">Architecture</span>
          </div>
        </motion.div>

        {/* BENTO BOX: RESNET50 */}
        <motion.div 
          whileHover={{ y: -5 }}
          className="border border-border bg-background p-6 flex flex-col group transition-all hover:border-primary/50"
        >
          <div className="bg-primary/5 p-4 w-fit mb-6">
            <BrainCircuit className="w-6 h-6 text-primary" />
          </div>
          <h3 className="font-display font-bold uppercase tracking-tighter text-xl mb-2">ResNet50</h3>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Our primary &quot;Heavyweight&quot; model. It features 50 layers of deep feature extraction, allowing it to see ultra-fine microscopic patterns that simple eyes might miss.
          </p>
        </motion.div>

        {/* BENTO BOX: GRAD-CAM */}
        <motion.div 
          whileHover={{ y: -5 }}
          className="border border-border bg-background p-6 flex flex-col group transition-all hover:border-primary/50"
        >
          <div className="bg-primary/5 p-4 w-fit mb-6">
            <Microscope className="w-6 h-6 text-primary" />
          </div>
          <h3 className="font-display font-bold uppercase tracking-tighter text-xl mb-2">Visual Logic</h3>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Standard AI is a &quot;Black Box&quot;. Grad-CAM (Heatmaps) forces the AI to highlight the exact pixels it used to make a decision—providing clinical proof and trust.
          </p>
        </motion.div>

        {/* BENTO BOX: FOCAL LOSS */}
        <motion.div 
          whileHover={{ scale: 1.02 }}
          className="lg:col-span-2 border-2 border-primary bg-primary/5 p-8 relative overflow-hidden group"
        >
          <ShieldAlert className="absolute -bottom-4 -right-4 w-32 h-32 opacity-5 text-primary rotate-12" />
          <h3 className="font-display font-black uppercase tracking-tighter text-4xl mb-6 leading-none">Solving Bias</h3>
          <p className="text-muted-foreground leading-relaxed max-w-md">
            Our data is imbalanced (some diseases are rare). Instead of ignoring rare spots, we use <span className="font-bold text-foreground">Focal Loss</span>. It mathematically &quot;shouts&quot; at the AI to pay 10x more attention to rare, hard cases than common ones.
          </p>
        </motion.div>

        {/* BENTO BOX: DATA CENTRIC */}
        <motion.div 
          whileHover={{ y: -5 }}
          className="border border-border bg-background p-6 flex flex-col group transition-all hover:border-primary/50"
        >
          <div className="bg-primary/5 p-4 w-fit mb-6">
            <GitBranch className="w-6 h-6 text-primary" />
          </div>
          <h3 className="font-display font-bold uppercase tracking-tighter text-xl mb-2">Data Centric</h3>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Instead of just refining the code, we refine the images. We use math to generate &quot;Synthetic Twins&quot; of rare lesions to give the AI more practice material.
          </p>
        </motion.div>

        {/* EXTERNAL LINK CARD */}
        <a 
          href="https://www.kaggle.com/datasets/nour12347653/skin-disease-detection-dataset-ham10000-isic" 
          target="_blank"
          className="border border-border bg-background p-6 flex items-center justify-between group hover:bg-foreground hover:text-background transition-all"
        >
          <div className="flex flex-col">
            <span className="font-display font-bold uppercase tracking-tighter text-xl">The Dataset</span>
            <span className="text-[10px] opacity-50">HAM10000 Repository</span>
          </div>
          <ExternalLink className="w-6 h-6 opacity-30 group-hover:opacity-100 transition-opacity" />
        </a>

      </div>

      <footer className="mt-20 text-center border-t border-border pt-12 opacity-50">
        <p className="text-xs tracking-widest uppercase font-bold">BCSE498J • Capstone Project Phase II Documentation</p>
      </footer>
    </div>
  );
}
