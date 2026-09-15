#!/usr/bin/env python3
from __future__ import annotations

"""
whitepaper_analyzer.py — DeFi Whitepaper Risk Due-Diligence Analyzer
Extracts text from whitepaper PDFs, chunks them with token counting,
analyzes each chunk for concrete risk signals using OpenAI (gpt-4o-mini),
synthesizes findings into a unified executive risk report, and renders
results both to terminal (via Rich) and Markdown/JSON files.
"""

import os
import json
import re
import time
import argparse
import warnings
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

from datetime import datetime
from pathlib import Path

from pypdf import PdfReader
from openai import OpenAI, RateLimitError
from pydantic import BaseModel, Field
from typing import Literal, List
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track
from rich.panel import Panel
import tiktoken

load_dotenv()
client = OpenAI()
console = Console()
enc = tiktoken.get_encoding("cl100k_base")

# ═══════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════

class Finding(BaseModel):
    category: str
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL"]
    description: str
    evidence_quote: str

class ChunkAnalysis(BaseModel):
    findings: List[Finding]
    section_summary: str

class TopRisk(BaseModel):
    category: str
    severity: str
    description: str

class FinalReport(BaseModel):
    overall_risk_rating: Literal["critical", "high", "medium", "low"]
    executive_summary: str
    top_risks: List[TopRisk]
    total_findings_analyzed: int
    document_name: str = ""
    generated_at: str = ""
    all_findings: List[Finding] = []


# ═══════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════

WHITEPAPER_CHUNK_ANALYSIS_SYSTEM = """You are a senior DeFi due-diligence analyst reviewing a 
section of a crypto project's whitepaper. Your job is to identify concrete risk signals — 
not to summarize the project's marketing claims.

Focus on these risk categories:
- TOKENOMICS: unclear vesting, excessive team allocation, inflationary mechanisms, unclear utility
- CENTRALIZATION: admin keys, upgradeable contracts without timelock/multisig, single points of control
- SECURITY: unaudited claims, vague security language, missing bug bounty mention
- ECONOMIC DESIGN: unsustainable yield promises, ponzi-like mechanics, unclear value accrual
- TRANSPARENCY: vague team info, anonymous founders without track record disclosure, missing audit links
- REGULATORY: securities-like token structure, unclear jurisdiction, unlicensed financial promises

Rules:
- Base findings STRICTLY on what is written in the provided text — do not assume anything not stated
- If a section contains no notable risk signals, say so explicitly — do not force findings to seem thorough
- Distinguish between "stated risk" and "inferred risk" (gaps or vague wording)
- Rate each finding's severity: CRITICAL, HIGH, MEDIUM, LOW, or INFORMATIONAL

Think through the section carefully before listing findings.

Respond in this format:
REASONING:
<brief analysis>

RESULT:
{"findings": [{"category": "...", "severity": "...", "description": "...", "evidence_quote": "..."}], "section_summary": "one sentence"}"""

WHITEPAPER_SYNTHESIS_SYSTEM = """You are a senior DeFi due-diligence analyst producing a final 
risk report by synthesizing findings from multiple whitepaper sections already analyzed.

Your job:
- Consolidate duplicate or overlapping findings
- Identify the TOP 5 most significant risks, ranked by severity and impact
- Give an overall risk rating: CRITICAL, HIGH, MEDIUM, or LOW
- Write an executive summary a busy investor could read in 30 seconds

Base synthesis STRICTLY on the findings provided. Do not introduce new categories not present 
in the input. If findings are sparse, say so honestly.

Respond in this format:
REASONING:
<how you consolidated and prioritized>

RESULT:
{"overall_risk_rating": "critical|high|medium|low", "executive_summary": "2-3 sentences", "top_risks": [{"category": "...", "severity": "...", "description": "..."}], "total_findings_analyzed": <number>}"""


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def extract_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text

def extract_reasoning_and_result(text: str) -> tuple[str, dict]:
    reasoning = ""
    if "RESULT:" in text:
        parts = text.split("RESULT:")
        reasoning = parts[0].replace("REASONING:", "").strip()
        result_raw = parts[1].strip()
    else:
        result_raw = text.strip()

    cleaned = extract_json(result_raw)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: extract the JSON object with regex
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            result = json.loads(match.group(0))
        else:
            raise ValueError(f"Could not parse JSON from response: {result_raw[:200]}")
            
    return reasoning, result

def count_tokens(text: str) -> int:
    return len(enc.encode(text))


# ═══════════════════════════════════════════════════════════
# STEP 1: PDF EXTRACTION
# ═══════════════════════════════════════════════════════════

def extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    full_text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text.append(page_text)
    return "\n\n".join(full_text)


# ═══════════════════════════════════════════════════════════
# STEP 2: CHUNKING
# ═══════════════════════════════════════════════════════════

def chunk_text(text: str, max_tokens: int = 2000) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for para in paragraphs:
        para_tokens = count_tokens(para)
        
        if para_tokens > max_tokens:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            chunks.append(para)
            continue
        
        if current_tokens + para_tokens > max_tokens:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [current_chunk[-1]] if current_chunk else []
            current_tokens = count_tokens(current_chunk[0]) if current_chunk else 0
        
        current_chunk.append(para)
        current_tokens += para_tokens
    
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks


# ═══════════════════════════════════════════════════════════
# STEP 3: PER-CHUNK ANALYSIS (OpenAI API with retries)
# ═══════════════════════════════════════════════════════════

def analyze_chunk(chunk_text: str, chunk_num: int, retries: int = 3, model: str = "gpt-4o-mini", mock: bool = False) -> ChunkAnalysis | None:
    if mock:
        return ChunkAnalysis(
            findings=[
                Finding(
                    category="CENTRALIZATION",
                    severity="HIGH",
                    description=f"Consensus vulnerability identified in section {chunk_num}: hash power concentration risk.",
                    evidence_quote="Any attacker controlling majority computing power can outpace honest nodes."
                ),
                Finding(
                    category="SECURITY",
                    severity="MEDIUM",
                    description="Confirmation latency and chain reorganization risk prior to settlement finality.",
                    evidence_quote="Nodes accept the longest proof-of-work chain as authoritative truth."
                )
            ],
            section_summary=f"Section {chunk_num} analysis of consensus mechanisms, incentive structures, and network propagation."
        )

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=800,
                temperature=0,
                messages=[
                    {"role": "system", "content": WHITEPAPER_CHUNK_ANALYSIS_SYSTEM},
                    {"role": "user", "content": chunk_text}
                ]
            )
            raw_text = response.choices[0].message.content
            _, result = extract_reasoning_and_result(raw_text)
            return ChunkAnalysis(**result)
        
        except RateLimitError as e:
            err_msg = str(e)
            if "insufficient_quota" in err_msg or "credit_balance_exhausted" in err_msg:
                console.print(f"  [bold red]OpenAI Quota Exhausted:[/bold red] {e.message}")
                console.print("  [yellow]Please add credits at: https://platform.openai.com/settings/organization/billing[/yellow]")
                return None
            wait = 2 ** attempt
            console.print(f"  [yellow]Rate limited on chunk {chunk_num}, waiting {wait}s...[/yellow]")
            time.sleep(wait)
        
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            console.print(f"  [red]Failed to parse chunk {chunk_num}: {e}[/red]")
            return None
        except Exception as e:
            console.print(f"  [red]API error on chunk {chunk_num}: {e}[/red]")
            return None
    
    console.print(f"  [red]Chunk {chunk_num} failed after {retries} retries[/red]")
    return None


# ═══════════════════════════════════════════════════════════
# STEP 4: SYNTHESIS (OpenAI API)
# ═══════════════════════════════════════════════════════════

def synthesize_report(all_findings: List[Finding], doc_name: str, model: str = "gpt-4o-mini", mock: bool = False) -> FinalReport:
    if mock:
        return FinalReport(
            overall_risk_rating="medium",
            executive_summary="The whitepaper outlines an innovative decentralized consensus architecture. While cryptographic principles are sound, key operational risks stem from majority hashrate concentration vulnerabilities and confirmation depth latency before finality.",
            top_risks=[
                TopRisk(category="CENTRALIZATION", severity="HIGH", description="51% mining power majority can theoretically rewrite recent block history and double-spend."),
                TopRisk(category="SECURITY", severity="MEDIUM", description="Settlement latency requires multiple confirmation blocks before economic finality is guaranteed."),
                TopRisk(category="ECONOMIC DESIGN", severity="MEDIUM", description="Long-term security budget transitions from block subsidies to transaction fees with unknown fee market stability.")
            ],
            total_findings_analyzed=len(all_findings),
            document_name=doc_name,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            all_findings=all_findings
        )

    findings_text = "\n".join([
        f"- [{f.severity}] [{f.category}] {f.description} (evidence: \"{f.evidence_quote[:100]}...\")"
        for f in all_findings
    ])
    
    response = client.chat.completions.create(
        model=model,
        max_tokens=1000,
        temperature=0,
        messages=[
            {"role": "system", "content": WHITEPAPER_SYNTHESIS_SYSTEM},
            {"role": "user", "content": f"Findings from all sections:\n\n{findings_text}"}
        ]
    )
    
    raw_text = response.choices[0].message.content
    _, result = extract_reasoning_and_result(raw_text)
    
    report = FinalReport(
        overall_risk_rating=result["overall_risk_rating"],
        executive_summary=result["executive_summary"],
        top_risks=[TopRisk(**r) for r in result["top_risks"]],
        total_findings_analyzed=len(all_findings),
        document_name=doc_name,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        all_findings=all_findings
    )
    return report


# ═══════════════════════════════════════════════════════════
# STEP 5: REPORT RENDERING
# ═══════════════════════════════════════════════════════════

SEVERITY_COLORS = {
    "CRITICAL": "bold red", "HIGH": "red", "MEDIUM": "yellow",
    "LOW": "blue", "INFORMATIONAL": "dim"
}

def print_report_to_terminal(report: FinalReport):
    console.print("\n")
    console.print(Panel(
        f"[bold]{report.document_name}[/bold]\n"
        f"Generated: {report.generated_at}  |  "
        f"Findings analyzed: {report.total_findings_analyzed}",
        title="DeFi Whitepaper Risk Analysis (OpenAI)",
        style="cyan"
    ))
    
    rating_color = {"critical": "bold red", "high": "red", "medium": "yellow", "low": "green"}
    console.print(f"\n[bold]Overall Risk Rating:[/bold] "
                  f"[{rating_color[report.overall_risk_rating]}]{report.overall_risk_rating.upper()}[/{rating_color[report.overall_risk_rating]}]\n")
    
    console.print(f"[bold]Executive Summary:[/bold]\n{report.executive_summary}\n")
    
    console.print("[bold]Top Risks:[/bold]")
    for i, risk in enumerate(report.top_risks, 1):
        color = SEVERITY_COLORS.get(risk.severity, "white")
        console.print(f"  {i}. [{color}]{risk.severity}[/{color}] [{risk.category}] {risk.description}")
    
    console.print(f"\n[dim]Full findings ({len(report.all_findings)} total) saved to output file.[/dim]\n")


def save_markdown_report(report: FinalReport, output_path: str):
    lines = [
        f"# Risk Analysis Report: {report.document_name}",
        f"*Generated: {report.generated_at}*",
        "",
        f"## Overall Risk Rating: {report.overall_risk_rating.upper()}",
        "",
        "## Executive Summary",
        report.executive_summary,
        "",
        "## Top Risks",
        ""
    ]
    for i, risk in enumerate(report.top_risks, 1):
        lines.append(f"{i}. **[{risk.severity}]** [{risk.category}] {risk.description}")
    
    lines += ["", "## All Findings (Detailed)", ""]
    for f in report.all_findings:
        lines.append(f"### [{f.severity}] {f.category}")
        lines.append(f"{f.description}")
        lines.append(f"> \"{f.evidence_quote}\"")
        lines.append("")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# ═══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════

def analyze_whitepaper(pdf_path: str, model: str = "gpt-4o-mini", mock: bool = False):
    doc_name = Path(pdf_path).stem
    
    mode_label = " [MOCK DEMO MODE]" if mock else ""
    console.print(f"\n[bold cyan]Analyzing:[/bold cyan] {pdf_path}{mode_label}\n")
    
    console.print("[1/4] Extracting text from PDF...")
    text = extract_pdf_text(pdf_path)
    total_tokens = count_tokens(text)
    console.print(f"  Extracted {len(text):,} characters (~{total_tokens:,} tokens)")
    
    console.print("\n[2/4] Splitting into chunks...")
    chunks = chunk_text(text, max_tokens=2000)
    console.print(f"  Split into {len(chunks)} chunks")
    
    console.print("\n[3/4] Analyzing each section for risk signals...")
    all_findings = []
    for i, chunk in enumerate(track(chunks, description="  Analyzing")):
        analysis = analyze_chunk(chunk, i + 1, model=model, mock=mock)
        if analysis:
            all_findings.extend(analysis.findings)
    
    console.print(f"  Found {len(all_findings)} total findings across all sections")
    
    if not all_findings:
        console.print("[yellow]No findings extracted — check the PDF content or API responses.[/yellow]")
        return
    
    console.print("\n[4/4] Synthesizing final report...")
    report = synthesize_report(all_findings, doc_name, model=model, mock=mock)
    
    print_report_to_terminal(report)
    
    # Save outputs
    output_json = f"{doc_name}_risk_report.json"
    output_md = f"{doc_name}_risk_report.md"
    
    with open(output_json, "w") as f:
        f.write(report.model_dump_json(indent=2))
    save_markdown_report(report, output_md)
    
    console.print(f"[green]Saved:[/green] {output_json}")
    console.print(f"[green]Saved:[/green] {output_md}")
    
    # Cost estimate
    estimated_calls = len(chunks) + 1
    call_type = "Simulated API calls (Mock)" if mock else f"API calls using {model}"
    console.print(f"\n[dim]Pipeline made {estimated_calls} {call_type}[/dim]")


# ═══════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze a DeFi/crypto whitepaper PDF for risk signals using OpenAI."
    )
    parser.add_argument("pdf_path", help="Path to the whitepaper PDF file")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use (default: gpt-4o-mini)")
    parser.add_argument("--mock", action="store_true", help="Run in mock demo mode without calling OpenAI API (free test)")
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        console.print(f"[red]Error: file not found: {args.pdf_path}[/red]")
        exit(1)
    
    analyze_whitepaper(args.pdf_path, model=args.model, mock=args.mock)
