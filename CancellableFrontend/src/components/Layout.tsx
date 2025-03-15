
import React from 'react';
import { cn } from '@/lib/utils';

interface LayoutProps {
  children: React.ReactNode;
  className?: string;
}

const Layout = ({ children, className }: LayoutProps) => {
  return (
    <div className={cn(
      "flex min-h-screen flex-col bg-background antialiased",
      className
    )}>
      <main className="flex-1">
        {children}
      </main>
    </div>
  );
};

export default Layout;
