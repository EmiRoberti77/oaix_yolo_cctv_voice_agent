"use client";

import { useEffect, useRef, useState } from "react";

export default function Home() {
  const imgRef = useRef<HTMLImageElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Use the API route to proxy the MJPEG stream
    const streamUrl = "/api/mjpeg";

    if (imgRef.current) {
      imgRef.current.src = streamUrl;

      imgRef.current.onload = () => {
        setIsLoading(false);
        setError(null);
      };

      imgRef.current.onerror = () => {
        setIsLoading(false);
        setError(
          "Failed to load MJPEG stream. Make sure the RTSP server is running on port 8000."
        );
      };
    }

    return () => {
      // Cleanup if needed
      if (imgRef.current) {
        imgRef.current.src = "";
      }
    };
  }, []);

  return (
    <main className="min-h-screen bg-gray-900 p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-white mb-6">
          CCTV MJPEG Stream
        </h1>

        {error && (
          <div className="bg-red-500 text-white p-4 rounded mb-4">{error}</div>
        )}

        {isLoading && !error && (
          <div className="bg-blue-500 text-white p-4 rounded mb-4">
            Loading stream...
          </div>
        )}

        <div className="bg-black rounded-lg overflow-hidden shadow-2xl">
          <img
            ref={imgRef}
            alt="MJPEG Stream"
            className="w-full h-auto"
            style={{
              display: error ? "none" : "block",
              maxHeight: "80vh",
              objectFit: "contain",
            }}
          />
        </div>

        <div className="mt-4 text-gray-400 text-sm">
          <p>
            Stream URL:{" "}
            <code className="bg-gray-800 px-2 py-1 rounded">/api/mjpeg</code>
          </p>
          <p className="mt-2">
            Make sure the RTSP server is running:{" "}
            <code className="bg-gray-800 px-2 py-1 rounded">
              python src/rtsp_server/app.py
            </code>
          </p>
        </div>
      </div>
    </main>
  );
}
