
import React, { useEffect, useRef } from 'react';
import { cn } from '@/lib/utils';
import { ScrollArea } from '@/components/ui/scroll-area';

export interface SimilarityEntry {
  timestamp: number;
  similarity: number;
}

interface SimilarityLogProps {
  entries: SimilarityEntry[];
  className?: string;
}

const SimilarityLog = ({ entries, className }: SimilarityLogProps) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [entries]);

  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
  };
  
  // Get similarity color based on percentage
  const getSimilarityColor = (similarity: number) => {
    if (similarity < 50) return 'text-red-500';
    if (similarity < 75) return 'text-yellow-500';
    return 'text-green-500';
  };

  return (
    <ScrollArea className={cn("h-[300px] rounded-xl border p-4", className)}>
      <div ref={scrollRef} className="space-y-1">
        <div className="flex justify-between mb-2 text-sm font-medium text-muted-foreground">
          <span>Timestamp</span>
          <span>Similarity</span>
        </div>
        
        {entries.length === 0 ? (
          <div className="flex h-32 items-center justify-center text-muted-foreground">
            <p>No similarity data available</p>
          </div>
        ) : (
          entries.map((entry, index) => (
            <div 
              key={index} 
              className={cn(
                "flex justify-between py-2 border-b border-border last:border-0",
                index === entries.length - 1 ? "animate-pulse bg-primary/5 rounded px-2 -mx-2" : ""
              )}
            >
              <span className="text-sm">{formatTime(entry.timestamp)}</span>
              <span className={cn(
                "text-sm font-medium", 
                getSimilarityColor(entry.similarity)
              )}>
                {entry.similarity.toFixed(2)}%
              </span>
            </div>
          ))
        )}
      </div>
    </ScrollArea>
  );
};

export default SimilarityLog;
