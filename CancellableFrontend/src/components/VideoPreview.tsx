
import React, { useRef, useEffect } from 'react';
import { cn } from '@/lib/utils';

interface VideoPreviewProps {
  file: File | null;
  className?: string;
}

const VideoPreview = ({ file, className }: VideoPreviewProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (file && videoRef.current) {
      const videoUrl = URL.createObjectURL(file);
      videoRef.current.src = videoUrl;
      
      return () => {
        URL.revokeObjectURL(videoUrl);
      };
    }
  }, [file]);

  if (!file) {
    return (
      <div className={cn(
        "w-full h-[250px] rounded-xl bg-muted flex items-center justify-center",
        className
      )}>
        <p className="text-muted-foreground">No video selected</p>
      </div>
    );
  }

  return (
    <video
      ref={videoRef}
      className={cn(
        "w-full h-auto rounded-xl object-cover bg-black max-h-[250px]",
        className
      )}
      controls
      muted
      playsInline
    />
  );
};

export default VideoPreview;
