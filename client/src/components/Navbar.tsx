'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Activity } from 'lucide-react';

export function Navbar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/90 backdrop-blur-md h-20">
      <div className="w-full max-w-7xl mx-auto xl:border-x xl:border-border h-full px-4 sm:px-6 flex items-center justify-between">
        
        {/* Brand */}
        <Link href="/" className="flex items-center gap-4 group">
          <div className="bg-primary text-primary-foreground p-2 shrink-0 transition-transform duration-500 ease-out group-hover:rotate-90">
            <Activity className="h-6 w-6" strokeWidth={2.5} />
          </div>
          <div className="flex flex-col">
            <span className="font-display font-bold text-xl leading-none tracking-tighter uppercase">Capstone <span className="text-primary">Project</span></span>
            <span className="text-[10px] text-muted-foreground uppercase tracking-widest mt-1 font-bold">BCSE498J</span>
          </div>
        </Link>
        
        {/* Navigation */}
        <nav className="hidden md:flex items-stretch h-full border-l border-border hover:border-l-primary/50 transition-colors">
          {[
            { name: 'Architecture', path: '/' },
            { name: 'Clinical Demo', path: '/demo' },
            { name: 'Metrics', path: '/dashboard' },
            { name: 'Glossary', path: '/knowledge' },
          ].map((item) => {
            const isActive = pathname === item.path;
            
            return (
              <Link
                key={item.path}
                href={item.path}
                className={`flex items-center px-6 text-xs font-bold tracking-[0.15em] uppercase transition-all border-r border-border ${
                  isActive
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:bg-white hover:text-black hover:border-r-white border-r-border'
                }`}
              >
                {item.name}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
