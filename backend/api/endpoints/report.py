import io
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from typing import Dict, Any
from app.utils.pdf_generator import PDFReportGenerator

router = APIRouter(prefix="/report", tags=["Report"])


@router.post("/generate")
def generate_report_endpoint(payload: Dict[str, Any]):
    try:
        pdf_bytes = PDFReportGenerator.create_pitwall_report(payload)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=TyreIQ_PitWall_Report.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation error: {str(e)}")
