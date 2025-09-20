"""
FastAPI web server for HTP (House-Tree-Person) analysis.
Provides REST API endpoints for image upload and psychological assessment.
"""

import os
import sys
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

# Load environment variables
load_dotenv('.env.local')

# Environment configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
MODEL_PATH = os.getenv("MODEL_PATH", "yolo11s.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.25))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10485760))  # 10MB default
TEMP_FILE_CLEANUP_HOURS = int(os.getenv("TEMP_FILE_CLEANUP_HOURS", 1))
STATIC_FILES_DIR = os.getenv("STATIC_FILES_DIR", "static")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
RELOAD = os.getenv("RELOAD", "False").lower() == "true"

# Parse allowed file types from environment
ALLOWED_FILE_TYPES_STR = os.getenv("ALLOWED_FILE_TYPES", "image/jpeg,image/png,image/jpg,image/bmp,image/tiff")
ALLOWED_FILE_TYPES = set(ALLOWED_FILE_TYPES_STR.split(","))

from feature_extractor import HTPFeatureExtractor

# Initialize FastAPI app
app = FastAPI(
    title="HTP Analyzer API",
    description="AI-powered House-Tree-Person psychological assessment analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
origins = [
    "http://localhost:3000",  # Next.js dev server
    "http://127.0.0.1:3000",  # Alternative localhost
    "https://htp-analyzer-backend.onrender.com",  # Backend Render domain
    "https://htp-test-analyzer.vercel.app",  # Specific frontend domain
]

# Get additional origins from environment
if additional_origins := os.getenv("ALLOWED_ORIGINS"):
    origins.extend(additional_origins.split(","))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,  # Cache preflight requests for 10 minutes
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # Don't set CSP as it's too restrictive for API
    return response

# Global variables
feature_extractor: Optional[HTPFeatureExtractor] = None
temp_files: Dict[str, Dict] = {}  # Track temporary files for cleanup

# Pydantic models for API responses
class AnalysisResult(BaseModel):
    analysis_id: str
    house_size_category: str
    detected_features: List[str]
    missing_features: List[str]
    risk_factors: List[str]
    positive_indicators: List[str]
    psychological_interpretation: str
    overall_confidence_score: float
    analysis_timestamp: datetime
    processing_time_seconds: float
    
    # Additional detailed analysis
    house_area_ratio: float
    house_placement: List[str]
    door_present: bool
    window_count: int
    chimney_present: bool
    detection_confidence: Dict[str, float]
    psychological_indicators: Dict[str, List[str]]

class AnalysisRequest(BaseModel):
    confidence_threshold: float = 0.25
    
class HealthResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool
    timestamp: datetime

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: datetime

# Mount static files for serving generated reports and visualizations
static_dir = Path(STATIC_FILES_DIR)
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_FILES_DIR), name="static")

@app.on_event("startup")
async def startup_event():
    """Initialize the ML model on startup."""
    global feature_extractor
    
    try:
        # Find the latest trained model
        model_path = find_latest_model()
        if not model_path:
            # Use a default model if no trained model exists
            model_path = MODEL_PATH
            print(f"⚠️ No trained model found, using default: {model_path}")
        else:
            print(f"✅ Loading trained model: {model_path}")
        
        feature_extractor = HTPFeatureExtractor(model_path)
        print("🚀 HTP Feature Extractor initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        feature_extractor = None

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    await cleanup_temp_files()

def find_latest_model():
    """Find the latest trained model."""
    training_outputs = Path("results/training/training_outputs")
    if not training_outputs.exists():
        return None

    # Look for the latest training run
    runs = list(training_outputs.glob("htp_yolo11s_*"))
    if not runs:
        return None

    # Get the most recent run
    latest_run = max(runs, key=lambda x: x.stat().st_mtime)
    model_path = latest_run / "weights" / "best.pt"

    return str(model_path) if model_path.exists() else None

async def cleanup_temp_files():
    """Clean up temporary files older than specified hours."""
    current_time = datetime.now()
    to_remove = []
    
    for file_id, file_info in temp_files.items():
        if current_time - file_info["created"] > timedelta(hours=TEMP_FILE_CLEANUP_HOURS):
            try:
                if os.path.exists(file_info["path"]):
                    os.remove(file_info["path"])
                to_remove.append(file_id)
            except Exception as e:
                print(f"⚠️ Failed to remove temp file: {e}")
    
    for file_id in to_remove:
        temp_files.pop(file_id, None)

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "HTP Analyzer API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if feature_extractor else "unhealthy",
        message="HTP Analyzer API is running" if feature_extractor else "Model not loaded",
        model_loaded=feature_extractor is not None,
        timestamp=datetime.now()
    )

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence_threshold: float = CONFIDENCE_THRESHOLD
):
    """
    Analyze a house drawing image and return psychological assessment.
    
    Args:
        file: House drawing image file (JPEG, PNG, etc.)
        confidence_threshold: Confidence threshold for object detection (0.0-1.0)
    
    Returns:
        Comprehensive analysis results including psychological interpretation
    """
    if not feature_extractor:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    # Validate confidence threshold
    if not 0.0 <= confidence_threshold <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="Confidence threshold must be between 0.0 and 1.0"
        )
    
    # Validate file type
    if file.content_type not in ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_FILE_TYPES)}"
        )
    
    # Validate file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.1f}MB."
        )
    
    # Reset file position
    await file.seek(0)
    
    # Generate unique analysis ID
    analysis_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        # Save uploaded file temporarily
        suffix = Path(file.filename).suffix if file.filename else ".jpg"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        
        async with aiofiles.open(temp_file.name, 'wb') as f:
            await f.write(content)
        
        # Track temp file for cleanup
        temp_files[analysis_id] = {
            "path": temp_file.name,
            "created": start_time
        }
        
        # Perform analysis
        analysis = feature_extractor.analyze_image(
            temp_file.name,
            confidence_threshold=confidence_threshold
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate overall confidence score
        overall_confidence = (
            sum(analysis.detection_confidence.values()) / len(analysis.detection_confidence)
            if analysis.detection_confidence else 0.0
        )
        
        # Create psychological interpretation summary
        interpretation_parts = []
        for category, indicators in analysis.psychological_indicators.items():
            if indicators:
                interpretation_parts.append(f"{category.replace('_', ' ').title()}: {', '.join(indicators[:2])}")
        
        psychological_interpretation = "; ".join(interpretation_parts) if interpretation_parts else "Standard developmental indicators observed"
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, analysis_id)
        
        # Return analysis results
        return AnalysisResult(
            analysis_id=analysis_id,
            house_size_category=analysis.house_size_category,
            detected_features=analysis.detected_features,
            missing_features=analysis.missing_features,
            risk_factors=analysis.risk_factors,
            positive_indicators=analysis.positive_indicators,
            psychological_interpretation=psychological_interpretation,
            overall_confidence_score=overall_confidence,
            analysis_timestamp=start_time,
            processing_time_seconds=processing_time,
            house_area_ratio=analysis.house_area_ratio,
            house_placement=analysis.house_placement,
            door_present=analysis.door_present,
            window_count=analysis.window_count,
            chimney_present=analysis.chimney_present,
            detection_confidence=analysis.detection_confidence,
            psychological_indicators=analysis.psychological_indicators
        )
        
    except Exception as e:
        # Cleanup on error
        background_tasks.add_task(cleanup_temp_file, analysis_id)
        
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/analyze-with-report")
async def analyze_with_report(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence_threshold: float = CONFIDENCE_THRESHOLD
):
    """
    Analyze image and generate downloadable report files.
    
    Returns:
        Analysis results with URLs to downloadable report and visualization
    """
    if not feature_extractor:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    # Validate confidence threshold
    if not 0.0 <= confidence_threshold <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="Confidence threshold must be between 0.0 and 1.0"
        )
    
    # Validate file type
    if file.content_type not in ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_FILE_TYPES)}"
        )
    
    # Validate file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.1f}MB."
        )
    
    analysis_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        # Save uploaded file temporarily
        suffix = Path(file.filename).suffix if file.filename else ".jpg"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        
        async with aiofiles.open(temp_file.name, 'wb') as f:
            await f.write(content)
        
        # Perform analysis
        analysis = feature_extractor.analyze_image(
            temp_file.name,
            confidence_threshold=confidence_threshold
        )
        
        # Generate report and visualization
        report_path = feature_extractor.generate_report(analysis, output_dir=STATIC_FILES_DIR)
        viz_path = feature_extractor.visualize_analysis(analysis, output_dir=STATIC_FILES_DIR)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate overall confidence score
        overall_confidence = (
            sum(analysis.detection_confidence.values()) / len(analysis.detection_confidence)
            if analysis.detection_confidence else 0.0
        )
        
        # Create psychological interpretation summary
        interpretation_parts = []
        for category, indicators in analysis.psychological_indicators.items():
            if indicators:
                interpretation_parts.append(f"{category.replace('_', ' ').title()}: {', '.join(indicators[:2])}")
        
        psychological_interpretation = "; ".join(interpretation_parts) if interpretation_parts else "Standard developmental indicators observed"
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file_path, temp_file.name)
        
        # Return results with file URLs
        return {
            "analysis_id": analysis_id,
            "house_size_category": analysis.house_size_category,
            "detected_features": analysis.detected_features,
            "missing_features": analysis.missing_features,
            "risk_factors": analysis.risk_factors,
            "positive_indicators": analysis.positive_indicators,
            "psychological_interpretation": psychological_interpretation,
            "overall_confidence_score": overall_confidence,
            "analysis_timestamp": start_time,
            "processing_time_seconds": processing_time,
            "house_area_ratio": analysis.house_area_ratio,
            "house_placement": analysis.house_placement,
            "door_present": analysis.door_present,
            "window_count": analysis.window_count,
            "chimney_present": analysis.chimney_present,
            "detection_confidence": analysis.detection_confidence,
            "psychological_indicators": analysis.psychological_indicators,
            "report_url": f"/static/{Path(report_path).name}" if report_path else None,
            "visualization_url": f"/static/{Path(viz_path).name}" if viz_path else None
        }
        
    except Exception as e:
        background_tasks.add_task(cleanup_temp_file_path, temp_file.name)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/download/{file_id}")
async def download_file(file_id: str):
    """Download generated report or visualization file."""
    file_path = static_dir / file_id
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=file_id,
        media_type='application/octet-stream'
    )

async def cleanup_temp_file(analysis_id: str):
    """Clean up a specific temporary file."""
    if analysis_id in temp_files:
        file_info = temp_files[analysis_id]
        try:
            if os.path.exists(file_info["path"]):
                os.remove(file_info["path"])
        except Exception as e:
            print(f"⚠️ Failed to remove temp file: {e}")
        finally:
            temp_files.pop(analysis_id, None)

async def cleanup_temp_file_path(file_path: str):
    """Clean up a specific file by path."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"⚠️ Failed to remove temp file: {e}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP {exc.status_code}",
            message=exc.detail,
            timestamp=datetime.now()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred",
            timestamp=datetime.now()
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    
    # Check if running in production
    is_production = os.getenv("RELOAD") == "False"
    
    if is_production:
        # Production server
        uvicorn.run(
            "app:app",
            host=HOST,
            port=PORT,
            reload=False,
            workers=1
        )
    else:
        # Development server - use environment variables
        uvicorn.run(
            "app:app",
            host=HOST,
            port=PORT,
            reload=RELOAD
        )