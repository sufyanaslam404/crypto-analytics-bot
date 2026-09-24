# smart_router/dashboard.py

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout

console = Console()

class RouterDashboard:
    def __init__(self):
        self.results = []
        self.decisions = []

    def log(self, decision, result):
        self.decisions.append(decision)
        self.results.append(result)

    def _build_summary_panel(self) -> Panel:
        total_actual = sum(r.actual_cost for r in self.results)
        total_baseline = sum(r.baseline_cost for r in self.results)
        total_savings = total_baseline - total_actual
        savings_pct = (total_savings / total_baseline * 100) if total_baseline > 0 else 0

        model_counts = {}
        for r in self.results:
            model_counts[r.model_used] = model_counts.get(r.model_used, 0) + 1

        model_breakdown = "  |  ".join(f"{m}: {c}" for m, c in model_counts.items())

        text = (
            f"[bold]Tasks processed:[/bold] {len(self.results)}\n"
            f"[bold]Actual cost:[/bold] ${total_actual:.5f}\n"
            f"[bold]Baseline cost (always o1-preview, no cache):[/bold] ${total_baseline:.5f}\n"
            f"[bold green]Savings: ${total_savings:.5f} ({savings_pct:.1f}%)[/bold green]\n"
            f"[dim]Model usage: {model_breakdown}[/dim]"
        )
        return Panel(text, title="Smart Router — Live Cost Summary", style="cyan")

    def _build_recent_table(self, n: int = 8) -> Table:
        table = Table(title=f"Last {n} Tasks")
        table.add_column("Task ID", style="dim")
        table.add_column("Model")
        table.add_column("Tokens (in/out)")
        table.add_column("Cache")
        table.add_column("Cost")
        table.add_column("Saved vs Baseline")

        for r in self.results[-n:]:
            cache_note = f"R:{r.cache_read_tokens} W:{r.cache_write_tokens}" if (r.cache_read_tokens or r.cache_write_tokens) else "-"
            table.add_row(
                r.task_id,
                r.model_used,
                f"{r.input_tokens}/{r.output_tokens}",
                cache_note,
                f"${r.actual_cost:.6f}",
                f"${r.savings:.6f}"
            )
        return table

    def render(self):
        console.print(self._build_summary_panel())
        console.print(self._build_recent_table())

    def print_final_report(self):
        console.print("\n")
        self.render()

        # Breakdown by task type
        by_type = {}
        for d, r in zip(self.decisions, self.results):
            by_type.setdefault(d.task_type, []).append(r)

        type_table = Table(title="Cost Breakdown by Task Type")
        type_table.add_column("Task Type")
        type_table.add_column("Count")
        type_table.add_column("Total Cost")
        type_table.add_column("Total Saved")

        for task_type, results in by_type.items():
            total_cost = sum(r.actual_cost for r in results)
            total_saved = sum(r.savings for r in results)
            type_table.add_row(task_type, str(len(results)), f"${total_cost:.5f}", f"${total_saved:.5f}")

        console.print(type_table)
