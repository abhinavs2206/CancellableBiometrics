
import { useLocation } from "react-router-dom";
import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { AlertTriangle } from "lucide-react";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname
    );
  }, [location.pathname]);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-background p-4">
      <div className="text-center max-w-md animate-fade-in">
        <div className="w-16 h-16 rounded-full bg-red-100 dark:bg-red-900/20 flex items-center justify-center mx-auto mb-6">
          <AlertTriangle className="h-8 w-8 text-red-600 dark:text-red-500" />
        </div>
        
        <h1 className="text-5xl font-bold mb-4">404</h1>
        <p className="text-xl text-muted-foreground mb-8">
          The page you're looking for couldn't be found.
        </p>
        
        <Button asChild size="lg" className="animate-slide-up">
          <a href="/" className="px-8">
            Return to Home
          </a>
        </Button>
      </div>
    </div>
  );
};

export default NotFound;
