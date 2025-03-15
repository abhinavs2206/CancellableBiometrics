
import React from 'react';
import { cn } from '@/lib/utils';

interface CircularProgressProps {
  value: number;
  size?: number;
  strokeWidth?: number;
  className?: string;
  showLabel?: boolean;
  labelClassName?: string;
}

const CircularProgress = ({
  value,
  size = 120,
  strokeWidth = 8,
  className,
  showLabel = true,
  labelClassName
}: CircularProgressProps) => {
  // Constrain the value between 0 and 100
  const percentage = Math.min(100, Math.max(0, value));
  
  // Circle calculations
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;
  
  // Determine color based on percentage
  const getColor = () => {
    if (percentage < 50) return 'stroke-red-500';
    if (percentage < 75) return 'stroke-yellow-500';
    return 'stroke-green-500';
  };

  return (
    <div className={cn("progress-ring-container flex items-center justify-center", className)}>
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        className="progress-ring"
      >
        <circle
          className="progress-ring__background"
          stroke="currentColor"
          strokeOpacity={0.2}
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth={strokeWidth}
        />
        <circle
          className={cn("progress-ring__progress", getColor())}
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </svg>
      {showLabel && (
        <div className={cn("absolute inset-0 flex items-center justify-center", labelClassName)}>
          <span className="text-2xl font-semibold">{Math.round(percentage)}%</span>
        </div>
      )}
    </div>
  );
};

export default CircularProgress;
