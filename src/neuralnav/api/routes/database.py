"""Database management API routes.

Provides endpoints for uploading benchmark data, checking DB status,
and resetting the benchmark database. These endpoints enable remote
management of NeuralNav deployments (e.g., on Kubernetes) without
needing shell access.
"""

import json
import logging
import os

import psycopg2
from fastapi import APIRouter, File, HTTPException, UploadFile

from neuralnav.knowledge_base.loader import get_db_stats, insert_benchmarks, reset_benchmarks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["database"])

_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:neuralnav@localhost:5432/neuralnav",
)


def _get_connection():
    """Get a database connection for DB management operations."""
    return psycopg2.connect(_DATABASE_URL)


@router.get("/db/status")
async def db_status():
    """Get current benchmark database statistics."""
    try:
        conn = _get_connection()
        try:
            stats = get_db_stats(conn)
            return {"success": True, **stats}
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Failed to get DB status: {e}")
        raise HTTPException(status_code=503, detail=f"Database not accessible: {e}") from e


@router.post("/db/upload-benchmarks")
async def upload_benchmarks(file: UploadFile = File(...)):
    """Upload a benchmark JSON file and load it into the database.

    The JSON file should have a top-level "benchmarks" array containing
    benchmark records. Duplicates (same model/hardware/traffic config)
    are silently skipped.

    Usage:
        curl -X POST -F 'file=@benchmarks.json' http://host/api/v1/db/upload-benchmarks
    """
    if not file.filename or not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="File must be a .json file")

    try:
        content = await file.read()
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    benchmarks = data.get("benchmarks", [])
    if not benchmarks:
        raise HTTPException(
            status_code=400,
            detail='No benchmarks found. JSON must have a top-level "benchmarks" array.',
        )

    try:
        conn = _get_connection()
        try:
            stats = insert_benchmarks(conn, benchmarks)
            logger.info(
                f"Uploaded {len(benchmarks)} benchmarks from {file.filename}, "
                f"DB now has {stats['total_benchmarks']} total"
            )
            return {
                "success": True,
                "filename": file.filename,
                "records_in_file": len(benchmarks),
                **stats,
            }
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Failed to load benchmarks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load benchmarks: {e}") from e


@router.post("/db/reset")
async def reset_database():
    """Reset the benchmark database by removing all benchmark data.

    This truncates the exported_summaries table (cascading to related tables).
    The schema is preserved — only data is removed.

    Usage:
        curl -X POST http://host/api/v1/db/reset
    """
    try:
        conn = _get_connection()
        try:
            reset_benchmarks(conn)
            stats = get_db_stats(conn)
            logger.info("Benchmark database reset via API")
            return {
                "success": True,
                "message": "Benchmark database has been reset",
                **stats,
            }
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Failed to reset database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset database: {e}") from e
