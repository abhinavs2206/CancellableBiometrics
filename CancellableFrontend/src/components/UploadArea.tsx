
import React, { useState, useRef } from 'react';
import { cn } from '@/lib/utils';
import { Upload, X } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface UploadAreaProps {
  onFileSelected: (file: File) => void;
  className?: string;
  label: string;
  acceptedFileTypes?: string;
  selectedFile?: File | null;
  onClearFile?: () => void;
}

const UploadArea = ({
  onFileSelected,
  className,
  label,
  acceptedFileTypes = "video/mp4,video/quicktime,video/webm",
  selectedFile,
  onClearFile
}: UploadAreaProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelected(e.target.files[0]);
    }
  };

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div
      className={cn(
        "relative rounded-xl p-6 transition-all duration-200",
        isDragging ? "bg-primary/10 border-primary" : "bg-secondary/50 border-border",
        "border-2 border-dashed hover:bg-secondary/80 cursor-pointer",
        className
      )}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleButtonClick}
    >
      <input
        type="file"
        className="hidden"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept={acceptedFileTypes}
      />
      
      <div className="flex flex-col items-center justify-center gap-4 text-center">
        {selectedFile ? (
          <>
            <div className="relative w-full">
              <Button 
                variant="ghost" 
                size="sm" 
                className="absolute -top-4 -right-4 rounded-full h-8 w-8 p-0"
                onClick={(e) => {
                  e.stopPropagation();
                  if (onClearFile) onClearFile();
                }}
              >
                <X className="h-4 w-4" />
              </Button>
              <div className="py-3 px-4 bg-background rounded-lg">
                <p className="font-medium text-sm truncate max-w-[200px]">{selectedFile.name}</p>
                <p className="text-xs text-muted-foreground">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>
          </>
        ) : (
          <>
            <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
              <Upload className="h-6 w-6 text-primary" />
            </div>
            <div>
              <p className="font-medium text-base">{label}</p>
              <p className="text-sm text-muted-foreground mt-1">
                Drag and drop or click to browse
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default UploadArea;
