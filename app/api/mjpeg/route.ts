import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const MJPEG_SERVER_URL =
  process.env.MJPEG_SERVER_URL || "http://localhost:8000/mjpeg";

export async function GET() {
  try {
    const response = await fetch(MJPEG_SERVER_URL, {
      cache: "no-store",
      headers: {
        "Cache-Control": "no-cache",
      },
    });

    if (!response.ok) {
      return new NextResponse("Failed to fetch MJPEG stream", {
        status: response.status,
      });
    }

    return new NextResponse(response.body, {
      headers: {
        "Content-Type": "multipart/x-mixed-replace; boundary=frame",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        Pragma: "no-cache",
        Expires: "0",
      },
    });
  } catch (error) {
    console.error("Error proxying MJPEG stream:", error);
    return new NextResponse("Error connecting to MJPEG server", {
      status: 500,
    });
  }
}
