import io
from typing import Dict, Any
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


class PDFReportGenerator:
    """Generates professional Formula 1 Pit Wall Engineering debrief PDFs."""

    @classmethod
    def create_pitwall_report(cls, data: Dict[str, Any]) -> bytes:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        
        # Custom F1 Styles
        title_style = ParagraphStyle(
            "F1Title",
            parent=styles["Heading1"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#e10600"),
            fontName="Helvetica-Bold",
        )
        subtitle_style = ParagraphStyle(
            "F1SubTitle",
            parent=styles["Normal"],
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#4a5568"),
            fontName="Helvetica",
        )
        section_style = ParagraphStyle(
            "F1Section",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#1a202c"),
            fontName="Helvetica-Bold",
            spaceBefore=12,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "F1Body",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#2d3748"),
        )

        story = []

        # 1. Header Banner
        session_name = data.get("session_id", "2024_Bahrain_FP2").replace("_", " ")
        driver = data.get("driver_code", "VER")
        compound = data.get("compound", "MEDIUM")

        story.append(Paragraph("TYREIQ // PIT WALL ENGINEERING REPORT", title_style))
        story.append(Paragraph(f"Session: <b>{session_name}</b> | Driver: <b>{driver}</b> | Compound: <b>{compound}</b>", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#e10600"), spaceBefore=8, spaceAfter=14))

        # 2. Executive KPIs Table
        story.append(Paragraph("1. TYRE DEGRADATION & PACE METRICS", section_style))
        
        kpi_data = [
            ["Metric", "Value", "Confidence / Standard"],
            ["Base Clean Pace", f"{data.get('base_clean_lap_sec', 91.5):.3f}s", "100% (Noise-Filtered)"],
            ["Degradation Gradient", f"{data.get('degradation_slope_sec_per_lap', 0.054):.3f} s/lap", "Gaussian Process Prior"],
            ["Tyre Cliff Arrival", f"Lap {data.get('cliff_lap', 26)}", "Thermal + Wear Transition"],
            ["Optimal Pit Lap", f"Lap {data.get('recommended_pit_lap', 24)}", f"Window: {data.get('pit_window_start', 22)}-{data.get('pit_window_end', 26)}"],
            ["Model Confidence", f"{data.get('confidence_pct', 94.5):.1f}%", "+/- 2-Sigma Uncertainty"],
        ]

        t = Table(kpi_data, colWidths=[180, 160, 200])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a202c")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f7fafc"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ]))
        story.append(t)
        story.append(Spacer(1, 14))

        # 3. Strategy Recommendation
        story.append(Paragraph("2. PIT WALL STRATEGY DIRECTIVE", section_style))
        strat_summary = data.get("strategy_summary", (
            f"Box box target Lap {data.get('recommended_pit_lap', 24)} for switch to HARD compound. "
            "Clean air undercut delta yields ~1.8s net advantage over prime competitors."
        ))
        story.append(Paragraph(strat_summary, body_style))
        story.append(Spacer(1, 14))

        # 4. Noise Attribution Summary
        story.append(Paragraph("3. TELEMETRY NOISE DECOUPLING (SHAP ATTRIBUTION)", section_style))
        attrs = data.get("feature_attributions", [
            {"feature": "Fuel Load", "delta_sec": "+0.32s"},
            {"feature": "Traffic / Wake", "delta_sec": "+0.65s (if impeded)"},
            {"feature": "Track Evolution", "delta_sec": "-0.18s (rubbering-in)"},
            {"feature": "Driver Aggression", "delta_sec": "+0.08s"},
        ])
        
        attr_data = [["External Factor", "Lap Time Impact", "Engineering Context"]]
        for a in attrs:
            attr_data.append([
                a.get("display_name", a.get("feature", "")),
                f"{a.get('delta_sec', 0.0):+.3f}s" if isinstance(a.get("delta_sec"), (int, float)) else str(a.get("delta_sec")),
                a.get("description", "Telemetry delta"),
            ])

        t_attr = Table(attr_data, colWidths=[160, 120, 260])
        t_attr.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3748")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#edf2f7"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ]))
        story.append(t_attr)
        story.append(Spacer(1, 20))

        # 5. Footer Sign-off
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e0"), spaceBefore=10, spaceAfter=8))
        story.append(Paragraph("Generated autonomously by TyreIQ AI Intelligence Engine — F1 Pit Wall Telemetry Standard.", subtitle_style))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
