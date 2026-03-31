import type { Metadata } from 'next';
import { Syne, DM_Sans } from 'next/font/google';
import './globals.css';
import { Navbar } from '@/components/Navbar';

const syne = Syne({ 
  subsets: ['latin'],
  variable: '--font-syne',
  display: 'swap',
});

const dmSans = DM_Sans({
  subsets: ['latin'],
  variable: '--font-dm-sans',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'Biotech Lesion Dashboard',
  description: 'High-end diagnostic architecture built with ResNet50.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${dmSans.variable} ${syne.variable} font-sans min-h-screen flex flex-col antialiased selection:bg-primary selection:text-foreground`}>
        <Navbar />
        <main className="flex-1 relative z-10 w-full max-w-7xl mx-auto xl:border-x xl:border-border min-h-[calc(100vh-80px)]">
          <div className="fixed inset-0 z-[-1] noise-bg opacity-30 mix-blend-overlay pointer-events-none" />
          <div className="fixed inset-0 z-[-2] grid-bg pointer-events-none" />
          {children}
        </main>
      </body>
    </html>
  );
}
