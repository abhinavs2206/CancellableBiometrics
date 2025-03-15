
import React, { useState, useRef, useEffect } from 'react';
import Layout from '@/components/Layout';
import UploadArea from '@/components/UploadArea';
import VideoPreview from '@/components/VideoPreview';
import CircularProgress from '@/components/CircularProgress';
import SimilarityLog, { SimilarityEntry } from '@/components/SimilarityLog';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { AlertCircle, Info, PlayCircle, StopCircle } from 'lucide-react';
import { toast } from '@/components/ui/use-toast';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

// Mock function to simulate face similarity processing
// In a real implementation, this would connect to your Python backend
const processFaceSimilarity = (referenceVideo: File, targetVideo: File, updateCallback: (timestamp: number, similarity: number) => void) => {
  let elapsed = 0;
  const interval = 500; // 500ms = half second
  
  // This is a mock implementation
  const mockProcess = setInterval(() => {
    elapsed += interval / 1000; // Convert to seconds
    
    // Generate a random similarity that generally increases over time
    // This is just for demo purposes
    const baseSimilarity = Math.min(95, 40 + elapsed * 5);
    const randomVariation = Math.random() * 10 - 5; // -5 to +5
    const similarity = Math.max(0, Math.min(100, baseSimilarity + randomVariation));
    
    updateCallback(elapsed, similarity);
    
    // Stop after 20 seconds (this would normally stop when the video ends)
    if (elapsed >= 20) {
      clearInterval(mockProcess);
    }
  }, interval);
  
  // Return a cleanup function
  return () => clearInterval(mockProcess);
};

const Index = () => {
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [targetFile, setTargetFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [similarityEntries, setSimilarityEntries] = useState<SimilarityEntry[]>([]);
  const [currentSimilarity, setCurrentSimilarity] = useState<number>(0);
  
  const processingRef = useRef<() => void | null>(null);
  
  // Clear all states
  const handleReset = () => {
    if (processingRef.current) {
      processingRef.current();
      processingRef.current = null;
    }
    setIsProcessing(false);
    setSimilarityEntries([]);
    setCurrentSimilarity(0);
  };

  // Handle reference file selection
  const handleReferenceFileSelected = (file: File) => {
    setReferenceFile(file);
    handleReset();
  };

  // Handle target file selection
  const handleTargetFileSelected = (file: File) => {
    setTargetFile(file);
    handleReset();
  };

  // Start processing
  const handleStartProcessing = () => {
    if (!referenceFile || !targetFile) {
      toast({
        title: "Missing videos",
        description: "Please select both reference and target videos.",
        variant: "destructive"
      });
      return;
    }

    setIsProcessing(true);
    setSimilarityEntries([]);
    setCurrentSimilarity(0);
    
    // Update callback that will be called at regular intervals
    const updateSimilarity = (timestamp: number, similarity: number) => {
      setSimilarityEntries(prev => [...prev, { timestamp, similarity }]);
      setCurrentSimilarity(similarity);
    };
    
    // Start the processing (mock implementation)
    const cleanup = processFaceSimilarity(referenceFile, targetFile, updateSimilarity);
    processingRef.current = cleanup;
    
    toast({
      title: "Processing started",
      description: "Analyzing face similarity between videos..."
    });
  };

  // Stop processing
  const handleStopProcessing = () => {
    if (processingRef.current) {
      processingRef.current();
      processingRef.current = null;
    }
    setIsProcessing(false);
    
    toast({
      title: "Processing stopped",
      description: "Face similarity analysis has been stopped."
    });
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (processingRef.current) {
        processingRef.current();
      }
    };
  }, []);

  return (
    <Layout>
      <div className="container max-w-6xl py-12 px-4 sm:px-6 space-y-8 animate-fade-in">
        {/* Header */}
        <div className="space-y-2 text-center">
          <Badge variant="outline" className="text-primary mb-2">Face Similarity Echo</Badge>
          <h1 className="text-4xl font-bold tracking-tight">Dynamic Face Similarity Analysis</h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Upload two videos to analyze face similarity over time with real-time percentage updates at half-second intervals.
          </p>
        </div>
        
        <Separator />
        
        {/* Main content */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Left column - Upload areas */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <span className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center text-primary">1</span>
                  Reference Video
                </CardTitle>
                <CardDescription>Upload a video containing the reference face</CardDescription>
              </CardHeader>
              <CardContent>
                <UploadArea
                  label="Upload Reference Video"
                  onFileSelected={handleReferenceFileSelected}
                  selectedFile={referenceFile}
                  onClearFile={() => setReferenceFile(null)}
                />
              </CardContent>
              <CardFooter>
                {referenceFile && (
                  <VideoPreview file={referenceFile} className="mt-4" />
                )}
              </CardFooter>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <span className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center text-primary">2</span>
                  Target Video
                </CardTitle>
                <CardDescription>Upload a video to compare against the reference</CardDescription>
              </CardHeader>
              <CardContent>
                <UploadArea
                  label="Upload Target Video"
                  onFileSelected={handleTargetFileSelected}
                  selectedFile={targetFile}
                  onClearFile={() => setTargetFile(null)}
                />
              </CardContent>
              <CardFooter>
                {targetFile && (
                  <VideoPreview file={targetFile} className="mt-4" />
                )}
              </CardFooter>
            </Card>
          </div>
          
          {/* Right column - Results */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <span className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center text-primary">3</span>
                  Analysis Controls
                </CardTitle>
                <CardDescription>Start or stop the face similarity analysis</CardDescription>
              </CardHeader>
              <CardContent className="flex justify-center gap-4">
                <Button
                  onClick={handleStartProcessing}
                  disabled={isProcessing || !referenceFile || !targetFile}
                  className="w-36"
                >
                  <PlayCircle className="mr-2 h-4 w-4" /> Start
                </Button>
                <Button
                  onClick={handleStopProcessing}
                  variant="outline"
                  disabled={!isProcessing}
                  className="w-36"
                >
                  <StopCircle className="mr-2 h-4 w-4" /> Stop
                </Button>
              </CardContent>
              <CardFooter>
                <Alert className="w-full">
                  <Info className="h-4 w-4" />
                  <AlertTitle>Processing details</AlertTitle>
                  <AlertDescription>
                    Face similarity is calculated at 0.5-second intervals and updated in real-time.
                  </AlertDescription>
                </Alert>
              </CardFooter>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Current Similarity</CardTitle>
                <CardDescription>Real-time face similarity percentage</CardDescription>
              </CardHeader>
              <CardContent className="flex justify-center">
                <CircularProgress 
                  value={currentSimilarity} 
                  size={180} 
                  strokeWidth={12}
                  className={isProcessing ? "animate-pulse" : ""}
                />
              </CardContent>
              <CardFooter className="flex justify-center">
                {currentSimilarity > 0 && (
                  <Badge variant={currentSimilarity >= 75 ? "default" : currentSimilarity >= 50 ? "outline" : "destructive"}>
                    {currentSimilarity >= 75 ? "PASS" : "FAIL"}
                  </Badge>
                )}
              </CardFooter>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Similarity Log</CardTitle>
                <CardDescription>History of similarity measurements over time</CardDescription>
              </CardHeader>
              <CardContent>
                <SimilarityLog entries={similarityEntries} />
              </CardContent>
            </Card>
          </div>
        </div>
        
        {/* Info section */}
        <div className="mt-12">
          <Alert variant="default" className="bg-primary/5 border-primary/20">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>How this would connect to your Python code</AlertTitle>
            <AlertDescription className="text-left">
              <p className="mb-2">
                In a full implementation, this UI would connect to a backend API that would:
              </p>
              <ol className="list-decimal pl-5 space-y-1">
                <li>Upload both videos to the server</li>
                <li>Process them with the Python MTCNN and OpenCV code</li>
                <li>Return similarity percentages at regular intervals (0.5 seconds)</li>
                <li>Stream the results back to this UI in real-time</li>
              </ol>
              <p className="mt-2">
                For now, this is demonstrating the UI with mock data to show how the dynamic similarity tracking would work.
              </p>
            </AlertDescription>
          </Alert>
        </div>
      </div>
    </Layout>
  );
};

export default Index;
