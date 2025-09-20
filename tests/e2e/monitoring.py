"""
Monitoring and Reporting Module for E2E Testing.

This module provides:
- Test execution monitoring
- Performance metrics collection
- Report generation and formatting
- Trend analysis capabilities
- Integration with external monitoring systems
"""
import json
import time
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging


@dataclass
class TestMetrics:
    """Test execution metrics."""
    test_name: str
    suite_name: str
    execution_time: float
    memory_peak_mb: float
    cpu_avg_percent: float
    success: bool
    timestamp: str
    error_message: Optional[str] = None
    coverage_percent: Optional[float] = None


@dataclass
class SuiteMetrics:
    """Test suite aggregated metrics."""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    total_duration: float
    avg_execution_time: float
    success_rate: float
    timestamp: str
    coverage_percent: Optional[float] = None


@dataclass
class TrendData:
    """Trend analysis data."""
    metric_name: str
    timestamps: List[str]
    values: List[float]
    trend_direction: str  # "improving", "degrading", "stable"
    change_percent: float


class TestMetricsCollector:
    """Collects and stores test execution metrics."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize metrics collector."""
        self.db_path = db_path or Path("test_metrics.db")
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create test metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name TEXT NOT NULL,
                    suite_name TEXT NOT NULL,
                    execution_time REAL NOT NULL,
                    memory_peak_mb REAL NOT NULL,
                    cpu_avg_percent REAL NOT NULL,
                    success BOOLEAN NOT NULL,
                    timestamp TEXT NOT NULL,
                    error_message TEXT,
                    coverage_percent REAL
                )
            """)
            
            # Create suite metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS suite_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    suite_name TEXT NOT NULL,
                    total_tests INTEGER NOT NULL,
                    passed_tests INTEGER NOT NULL,
                    failed_tests INTEGER NOT NULL,
                    skipped_tests INTEGER NOT NULL,
                    total_duration REAL NOT NULL,
                    avg_execution_time REAL NOT NULL,
                    success_rate REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    coverage_percent REAL
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_timestamp ON test_metrics(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_suite_timestamp ON suite_metrics(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_suite ON test_metrics(suite_name)")
            
            conn.commit()
    
    def record_test_metrics(self, metrics: TestMetrics):
        """Record individual test metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO test_metrics (
                    test_name, suite_name, execution_time, memory_peak_mb,
                    cpu_avg_percent, success, timestamp, error_message, coverage_percent
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.test_name, metrics.suite_name, metrics.execution_time,
                metrics.memory_peak_mb, metrics.cpu_avg_percent, metrics.success,
                metrics.timestamp, metrics.error_message, metrics.coverage_percent
            ))
            conn.commit()
    
    def record_suite_metrics(self, metrics: SuiteMetrics):
        """Record test suite metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO suite_metrics (
                    suite_name, total_tests, passed_tests, failed_tests, skipped_tests,
                    total_duration, avg_execution_time, success_rate, timestamp, coverage_percent
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.suite_name, metrics.total_tests, metrics.passed_tests,
                metrics.failed_tests, metrics.skipped_tests, metrics.total_duration,
                metrics.avg_execution_time, metrics.success_rate, metrics.timestamp,
                metrics.coverage_percent
            ))
            conn.commit()
    
    def get_suite_history(self, suite_name: str, days: int = 30) -> List[SuiteMetrics]:
        """Get historical data for a test suite."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM suite_metrics 
                WHERE suite_name = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """, (suite_name, cutoff_date))
            
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            return [
                SuiteMetrics(**dict(zip(columns[1:], row[1:])))  # Skip ID column
                for row in rows
            ]
    
    def get_performance_trends(self, suite_name: str, days: int = 30) -> Dict[str, TrendData]:
        """Analyze performance trends for a test suite."""
        history = self.get_suite_history(suite_name, days)
        
        if len(history) < 2:
            return {}
        
        # Sort by timestamp
        history.sort(key=lambda x: x.timestamp)
        
        trends = {}
        
        # Analyze execution time trend
        execution_times = [h.avg_execution_time for h in history]
        timestamps = [h.timestamp for h in history]
        
        trends['execution_time'] = self._calculate_trend(
            'avg_execution_time', timestamps, execution_times
        )
        
        # Analyze success rate trend
        success_rates = [h.success_rate for h in history]
        trends['success_rate'] = self._calculate_trend(
            'success_rate', timestamps, success_rates
        )
        
        # Analyze coverage trend (if available)
        coverage_values = [h.coverage_percent for h in history if h.coverage_percent is not None]
        if coverage_values:
            coverage_timestamps = [h.timestamp for h in history if h.coverage_percent is not None]
            trends['coverage'] = self._calculate_trend(
                'coverage', coverage_timestamps, coverage_values
            )
        
        return trends
    
    def _calculate_trend(self, metric_name: str, timestamps: List[str], values: List[float]) -> TrendData:
        """Calculate trend direction and change percentage."""
        if len(values) < 2:
            return TrendData(metric_name, timestamps, values, "stable", 0.0)
        
        # Simple linear trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        change_percent = ((second_avg - first_avg) / first_avg) * 100 if first_avg != 0 else 0
        
        # Determine trend direction
        if abs(change_percent) < 5:  # Less than 5% change
            trend_direction = "stable"
        elif change_percent > 0:
            # For success rate and coverage, positive is improving
            # For execution time, positive is degrading
            if metric_name in ['success_rate', 'coverage']:
                trend_direction = "improving"
            else:
                trend_direction = "degrading"
        else:
            if metric_name in ['success_rate', 'coverage']:
                trend_direction = "degrading"
            else:
                trend_direction = "improving"
        
        return TrendData(metric_name, timestamps, values, trend_direction, change_percent)


class ReportGenerator:
    """Generates various types of test reports."""
    
    def __init__(self, metrics_collector: TestMetricsCollector):
        """Initialize report generator."""
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)
    
    def generate_html_report(self, output_path: Path, suite_name: Optional[str] = None):
        """Generate HTML report with charts and metrics."""
        html_content = self._build_html_report(suite_name)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report generated: {output_path}")
    
    def generate_json_report(self, output_path: Path, suite_name: Optional[str] = None) -> Dict:
        """Generate JSON report with metrics data."""
        report_data = self._build_report_data(suite_name)
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"JSON report generated: {output_path}")
        return report_data
    
    def generate_markdown_report(self, output_path: Path, suite_name: Optional[str] = None):
        """Generate Markdown report for documentation."""
        markdown_content = self._build_markdown_report(suite_name)
        
        with open(output_path, 'w') as f:
            f.write(markdown_content)
        
        self.logger.info(f"Markdown report generated: {output_path}")
    
    def _build_report_data(self, suite_name: Optional[str] = None) -> Dict:
        """Build report data structure."""
        if suite_name:
            # Single suite report
            history = self.metrics_collector.get_suite_history(suite_name, days=30)
            trends = self.metrics_collector.get_performance_trends(suite_name, days=30)
            
            return {
                "suite_name": suite_name,
                "generated_at": datetime.now().isoformat(),
                "history": [asdict(h) for h in history],
                "trends": {k: asdict(v) for k, v in trends.items()},
                "latest_metrics": asdict(history[0]) if history else None
            }
        else:
            # All suites report
            with sqlite3.connect(self.metrics_collector.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT suite_name FROM suite_metrics
                    ORDER BY suite_name
                """)
                suite_names = [row[0] for row in cursor.fetchall()]
            
            suites_data = {}
            for name in suite_names:
                history = self.metrics_collector.get_suite_history(name, days=7)  # Last week
                trends = self.metrics_collector.get_performance_trends(name, days=7)
                
                suites_data[name] = {
                    "latest_metrics": asdict(history[0]) if history else None,
                    "trend_summary": self._summarize_trends(trends)
                }
            
            return {
                "generated_at": datetime.now().isoformat(),
                "suites": suites_data,
                "summary": self._calculate_overall_summary(suites_data)
            }
    
    def _build_html_report(self, suite_name: Optional[str] = None) -> str:
        """Build HTML report content."""
        report_data = self._build_report_data(suite_name)
        
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E2E Test Report - {title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1, h2, h3 {{ color: #333; }}
        .metric-card {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }}
        .success {{ border-color: #28a745; }}
        .warning {{ border-color: #ffc107; }}
        .danger {{ border-color: #dc3545; }}
        .trend-up {{ color: #28a745; }}
        .trend-down {{ color: #dc3545; }}
        .trend-stable {{ color: #6c757d; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; }}
        .chart-placeholder {{ background: #e9ecef; padding: 40px; text-align: center; margin: 20px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>E2E Test Report</h1>
        <p><strong>Generated:</strong> {generated_at}</p>
        
        {content}
    </div>
</body>
</html>
        """
        
        if suite_name:
            content = self._build_suite_html_content(report_data)
            title = f"{suite_name} Suite"
        else:
            content = self._build_overview_html_content(report_data)
            title = "Overview"
        
        return html_template.format(
            title=title,
            generated_at=report_data["generated_at"],
            content=content
        )
    
    def _build_suite_html_content(self, data: Dict) -> str:
        """Build HTML content for single suite report."""
        latest = data.get("latest_metrics")
        if not latest:
            return "<p>No metrics available for this suite.</p>"
        
        trends_html = ""
        for trend_name, trend_data in data.get("trends", {}).items():
            direction_class = f"trend-{trend_data['trend_direction'].replace('degrading', 'down').replace('improving', 'up')}"
            trends_html += f"""
            <div class="metric-card">
                <h4>{trend_name.replace('_', ' ').title()}</h4>
                <p class="{direction_class}">
                    {trend_data['trend_direction'].title()} 
                    ({trend_data['change_percent']:+.1f}%)
                </p>
            </div>
            """
        
        return f"""
        <h2>Latest Metrics</h2>
        <div class="metric-card {'success' if latest['success_rate'] > 90 else 'warning' if latest['success_rate'] > 70 else 'danger'}">
            <h3>Success Rate: {latest['success_rate']:.1f}%</h3>
            <p>Passed: {latest['passed_tests']}, Failed: {latest['failed_tests']}, Skipped: {latest['skipped_tests']}</p>
        </div>
        
        <div class="metric-card">
            <h3>Performance</h3>
            <p>Average Execution Time: {latest['avg_execution_time']:.2f}s</p>
            <p>Total Duration: {latest['total_duration']:.2f}s</p>
        </div>
        
        <h2>Trends (Last 30 Days)</h2>
        {trends_html}
        
        <div class="chart-placeholder">
            Performance Charts Would Appear Here
            <br><small>(Chart.js integration would be added for production)</small>
        </div>
        """
    
    def _build_overview_html_content(self, data: Dict) -> str:
        """Build HTML content for overview report."""
        suites_html = ""
        
        for suite_name, suite_data in data.get("suites", {}).items():
            latest = suite_data.get("latest_metrics")
            if not latest:
                continue
            
            status_class = 'success' if latest['success_rate'] > 90 else 'warning' if latest['success_rate'] > 70 else 'danger'
            
            suites_html += f"""
            <tr>
                <td>{suite_name}</td>
                <td class="{status_class}">{latest['success_rate']:.1f}%</td>
                <td>{latest['total_tests']}</td>
                <td>{latest['avg_execution_time']:.2f}s</td>
                <td>{latest['timestamp'][:10]}</td>
            </tr>
            """
        
        return f"""
        <h2>Test Suites Overview</h2>
        <table>
            <thead>
                <tr>
                    <th>Suite Name</th>
                    <th>Success Rate</th>
                    <th>Total Tests</th>
                    <th>Avg Duration</th>
                    <th>Last Run</th>
                </tr>
            </thead>
            <tbody>
                {suites_html}
            </tbody>
        </table>
        
        <h2>Summary</h2>
        {self._build_summary_html(data.get('summary', {}))}
        """
    
    def _build_summary_html(self, summary: Dict) -> str:
        """Build summary HTML section."""
        if not summary:
            return "<p>No summary data available.</p>"
        
        return f"""
        <div class="metric-card">
            <h3>Overall Health</h3>
            <p>Average Success Rate: {summary.get('avg_success_rate', 0):.1f}%</p>
            <p>Total Test Suites: {summary.get('total_suites', 0)}</p>
            <p>Suites Passing: {summary.get('passing_suites', 0)}</p>
        </div>
        """
    
    def _build_markdown_report(self, suite_name: Optional[str] = None) -> str:
        """Build Markdown report content."""
        report_data = self._build_report_data(suite_name)
        
        if suite_name:
            return self._build_suite_markdown_content(report_data)
        else:
            return self._build_overview_markdown_content(report_data)
    
    def _build_suite_markdown_content(self, data: Dict) -> str:
        """Build Markdown content for single suite report."""
        latest = data.get("latest_metrics")
        if not latest:
            return "# No Metrics Available\n\nNo metrics data found for this suite.\n"
        
        trends_md = ""
        for trend_name, trend_data in data.get("trends", {}).items():
            direction_emoji = "📈" if trend_data['trend_direction'] == "improving" else "📉" if trend_data['trend_direction'] == "degrading" else "📊"
            trends_md += f"- **{trend_name.replace('_', ' ').title()}**: {direction_emoji} {trend_data['trend_direction'].title()} ({trend_data['change_percent']:+.1f}%)\n"
        
        return f"""# E2E Test Report - {data['suite_name']}

**Generated:** {data['generated_at']}

## Latest Metrics

### Success Rate: {latest['success_rate']:.1f}%
- ✅ Passed: {latest['passed_tests']}
- ❌ Failed: {latest['failed_tests']}
- ⏭️ Skipped: {latest['skipped_tests']}

### Performance
- **Average Execution Time:** {latest['avg_execution_time']:.2f}s
- **Total Duration:** {latest['total_duration']:.2f}s
- **Last Run:** {latest['timestamp']}

## Trends (Last 30 Days)

{trends_md}

## Recommendations

{self._generate_recommendations(latest, data.get('trends', {}))}
"""
    
    def _build_overview_markdown_content(self, data: Dict) -> str:
        """Build Markdown content for overview report."""
        suites_table = "| Suite Name | Success Rate | Total Tests | Avg Duration | Last Run |\n|------------|--------------|-------------|--------------|----------|\n"
        
        for suite_name, suite_data in data.get("suites", {}).items():
            latest = suite_data.get("latest_metrics")
            if not latest:
                continue
            
            status_emoji = "✅" if latest['success_rate'] > 90 else "⚠️" if latest['success_rate'] > 70 else "❌"
            
            suites_table += f"| {suite_name} | {status_emoji} {latest['success_rate']:.1f}% | {latest['total_tests']} | {latest['avg_execution_time']:.2f}s | {latest['timestamp'][:10]} |\n"
        
        summary = data.get('summary', {})
        
        return f"""# E2E Test Report - Overview

**Generated:** {data['generated_at']}

## Test Suites Status

{suites_table}

## Summary

- **Total Test Suites:** {summary.get('total_suites', 0)}
- **Suites Passing (>90%):** {summary.get('passing_suites', 0)}
- **Average Success Rate:** {summary.get('avg_success_rate', 0):.1f}%

## Overall Health: {self._get_health_status(summary)}

{self._generate_overview_recommendations(data.get('suites', {}))}
"""
    
    def _summarize_trends(self, trends: Dict[str, TrendData]) -> Dict:
        """Summarize trend data for overview."""
        summary = {}
        for trend_name, trend_data in trends.items():
            summary[trend_name] = {
                "direction": trend_data.trend_direction,
                "change_percent": trend_data.change_percent
            }
        return summary
    
    def _calculate_overall_summary(self, suites_data: Dict) -> Dict:
        """Calculate overall summary statistics."""
        total_suites = len(suites_data)
        passing_suites = 0
        success_rates = []
        
        for suite_data in suites_data.values():
            latest = suite_data.get("latest_metrics")
            if latest:
                success_rates.append(latest['success_rate'])
                if latest['success_rate'] > 90:
                    passing_suites += 1
        
        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
        
        return {
            "total_suites": total_suites,
            "passing_suites": passing_suites,
            "avg_success_rate": avg_success_rate
        }
    
    def _generate_recommendations(self, latest_metrics: Dict, trends: Dict) -> str:
        """Generate recommendations based on metrics and trends."""
        recommendations = []
        
        # Success rate recommendations
        if latest_metrics['success_rate'] < 90:
            recommendations.append("🚨 Success rate is below 90%. Investigate and fix failing tests.")
        
        # Performance recommendations
        execution_trend = trends.get('execution_time')
        if execution_trend and execution_trend['trend_direction'] == 'degrading':
            recommendations.append("⚠️ Execution time is increasing. Consider performance optimization.")
        
        # Coverage recommendations
        if latest_metrics.get('coverage_percent') and latest_metrics['coverage_percent'] < 80:
            recommendations.append("📝 Code coverage is below 80%. Add more comprehensive tests.")
        
        if not recommendations:
            recommendations.append("✅ All metrics look good! Keep up the excellent work.")
        
        return "\n".join(f"- {rec}" for rec in recommendations)
    
    def _generate_overview_recommendations(self, suites_data: Dict) -> str:
        """Generate overview recommendations."""
        failing_suites = []
        
        for suite_name, suite_data in suites_data.items():
            latest = suite_data.get("latest_metrics")
            if latest and latest['success_rate'] < 80:
                failing_suites.append(suite_name)
        
        recommendations = []
        
        if failing_suites:
            recommendations.append(f"🚨 **Critical:** The following suites need immediate attention: {', '.join(failing_suites)}")
        
        recommendations.append("📊 Regular monitoring helps maintain test quality and catch regressions early.")
        recommendations.append("🔄 Consider setting up automated alerts for significant metric changes.")
        
        return "## Recommendations\n\n" + "\n".join(f"- {rec}" for rec in recommendations)
    
    def _get_health_status(self, summary: Dict) -> str:
        """Get overall health status."""
        avg_success_rate = summary.get('avg_success_rate', 0)
        
        if avg_success_rate >= 95:
            return "🟢 Excellent"
        elif avg_success_rate >= 85:
            return "🟡 Good"
        elif avg_success_rate >= 70:
            return "🟠 Needs Attention"
        else:
            return "🔴 Critical"


class AlertSystem:
    """Alert system for test failures and performance degradation."""
    
    def __init__(self, metrics_collector: TestMetricsCollector):
        """Initialize alert system."""
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)
    
    def check_and_send_alerts(self, suite_name: str, latest_metrics: SuiteMetrics):
        """Check metrics and send alerts if thresholds are exceeded."""
        alerts = []
        
        # Success rate alert
        if latest_metrics.success_rate < 80:
            alerts.append({
                "level": "critical",
                "message": f"Success rate dropped to {latest_metrics.success_rate:.1f}% in {suite_name}"
            })
        elif latest_metrics.success_rate < 95:
            alerts.append({
                "level": "warning", 
                "message": f"Success rate is {latest_metrics.success_rate:.1f}% in {suite_name}"
            })
        
        # Performance degradation alert
        trends = self.metrics_collector.get_performance_trends(suite_name, days=7)
        execution_trend = trends.get('execution_time')
        
        if execution_trend and execution_trend.trend_direction == 'degrading' and execution_trend.change_percent > 20:
            alerts.append({
                "level": "warning",
                "message": f"Execution time increased by {execution_trend.change_percent:.1f}% in {suite_name}"
            })
        
        # Send alerts
        for alert in alerts:
            self._send_alert(alert)
    
    def _send_alert(self, alert: Dict):
        """Send alert (placeholder for integration with notification systems)."""
        self.logger.warning(f"ALERT [{alert['level'].upper()}]: {alert['message']}")
        
        # Here you would integrate with:
        # - Slack notifications
        # - Email alerts
        # - PagerDuty
        # - Discord webhooks
        # - etc.


# Example usage and integration
def monitor_test_execution(test_results: Dict, output_dir: Path):
    """Monitor test execution and generate reports."""
    metrics_collector = TestMetricsCollector(output_dir / "test_metrics.db")
    report_generator = ReportGenerator(metrics_collector)
    alert_system = AlertSystem(metrics_collector)
    
    # Record metrics for each suite
    for suite_name, suite_result in test_results.items():
        # Convert results to metrics format
        suite_metrics = SuiteMetrics(
            suite_name=suite_name,
            total_tests=suite_result.get('total', 0),
            passed_tests=suite_result.get('passed', 0),
            failed_tests=suite_result.get('failed', 0),
            skipped_tests=suite_result.get('skipped', 0),
            total_duration=suite_result.get('duration', 0),
            avg_execution_time=suite_result.get('avg_time', 0),
            success_rate=suite_result.get('success_rate', 0),
            timestamp=datetime.now().isoformat(),
            coverage_percent=suite_result.get('coverage', None)
        )
        
        # Record metrics
        metrics_collector.record_suite_metrics(suite_metrics)
        
        # Check for alerts
        alert_system.check_and_send_alerts(suite_name, suite_metrics)
    
    # Generate reports
    report_generator.generate_html_report(output_dir / "test_report.html")
    report_generator.generate_json_report(output_dir / "test_report.json") 
    report_generator.generate_markdown_report(output_dir / "test_report.md")
    
    return metrics_collector, report_generator