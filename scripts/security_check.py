#!/usr/bin/env python3
"""Security check script to validate critical fixes have been applied.

Run this script to verify all critical security issues have been addressed
before deploying to production.
"""

import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ANSI color codes for terminal output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


class SecurityChecker:
    """Check for security vulnerabilities in the codebase."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_dir = project_root / "src"
        self.issues: List[Dict] = []
        self.warnings: List[Dict] = []
        
    def run_all_checks(self) -> Tuple[bool, List[Dict], List[Dict]]:
        """Run all security checks.
        
        Returns:
            Tuple of (all_passed, critical_issues, warnings)
        """
        print(f"{BLUE}Starting Security Audit...{RESET}\n")
        
        # Run checks
        self.check_sql_injection()
        self.check_pickle_usage()
        self.check_subprocess_shell()
        self.check_hardcoded_secrets()
        self.check_exception_handling()
        self.check_resource_cleanup()
        
        # Summary
        all_passed = len(self.issues) == 0
        return all_passed, self.issues, self.warnings
    
    def check_sql_injection(self) -> None:
        """Check for SQL injection vulnerabilities."""
        print(f"Checking for SQL injection vulnerabilities...")
        
        sql_patterns = [
            r'f".*SELECT.*WHERE.*{',  # f-string in SQL
            r"f'.*SELECT.*WHERE.*{",  # f-string in SQL
            r'%.*SELECT.*WHERE.*%',   # % formatting in SQL
            r'\.format\(.*SELECT.*WHERE',  # .format() in SQL
            r'\+.*SELECT.*WHERE.*\+',  # String concatenation in SQL
        ]
        
        for py_file in self.src_dir.rglob("*.py"):
            content = py_file.read_text()
            for pattern in sql_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    self.issues.append({
                        'type': 'SQL_INJECTION',
                        'file': str(py_file.relative_to(self.project_root)),
                        'severity': 'CRITICAL',
                        'message': 'Potential SQL injection vulnerability'
                    })
                    break
        
        self._print_check_result("SQL Injection", 'SQL_INJECTION')
    
    def check_pickle_usage(self) -> None:
        """Check for unsafe pickle usage."""
        print(f"Checking for unsafe pickle usage...")
        
        pickle_patterns = [
            r'pickle\.loads?\(',
            r'pickle\.load\(',
            r'cPickle\.loads?\(',
            r'dill\.loads?\(',
        ]
        
        for py_file in self.src_dir.rglob("*.py"):
            content = py_file.read_text()
            for pattern in pickle_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    # Check if it's deserializing untrusted data
                    line_num = content[:match.start()].count('\n') + 1
                    if not self._is_pickle_safe(py_file, line_num):
                        self.issues.append({
                            'type': 'UNSAFE_PICKLE',
                            'file': str(py_file.relative_to(self.project_root)),
                            'line': line_num,
                            'severity': 'CRITICAL',
                            'message': 'Unsafe pickle deserialization'
                        })
        
        self._print_check_result("Pickle Usage", 'UNSAFE_PICKLE')
    
    def check_subprocess_shell(self) -> None:
        """Check for subprocess with shell=True."""
        print(f"Checking for unsafe subprocess calls...")
        
        for py_file in self.src_dir.rglob("*.py"):
            content = py_file.read_text()
            
            # Check for shell=True
            if 'shell=True' in content:
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if 'shell=True' in line:
                        self.issues.append({
                            'type': 'SHELL_INJECTION',
                            'file': str(py_file.relative_to(self.project_root)),
                            'line': i,
                            'severity': 'HIGH',
                            'message': 'subprocess with shell=True is dangerous'
                        })
            
            # Check for os.system
            if 'os.system(' in content:
                self.issues.append({
                    'type': 'SHELL_INJECTION',
                    'file': str(py_file.relative_to(self.project_root)),
                    'severity': 'HIGH',
                    'message': 'os.system() is dangerous, use subprocess'
                })
        
        self._print_check_result("Subprocess Security", 'SHELL_INJECTION')
    
    def check_hardcoded_secrets(self) -> None:
        """Check for hardcoded secrets."""
        print(f"Checking for hardcoded secrets...")
        
        secret_patterns = [
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'password\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'["\']sk-[a-zA-Z0-9]{40,}["\']',  # API keys
            r'["\']AIza[a-zA-Z0-9_-]{35}["\']',  # Google API keys
        ]
        
        for py_file in self.src_dir.rglob("*.py"):
            content = py_file.read_text()
            for pattern in secret_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    # Check if it's a real secret or just a placeholder
                    if not any(placeholder in content for placeholder in 
                              ['your-api-key', 'your_api_key', '<api_key>', 
                               'example', 'placeholder', 'getenv', 'environ']):
                        self.warnings.append({
                            'type': 'HARDCODED_SECRET',
                            'file': str(py_file.relative_to(self.project_root)),
                            'severity': 'HIGH',
                            'message': 'Possible hardcoded secret'
                        })
        
        self._print_check_result("Hardcoded Secrets", 'HARDCODED_SECRET', check_warnings=True)
    
    def check_exception_handling(self) -> None:
        """Check for poor exception handling."""
        print(f"Checking exception handling...")
        
        bad_patterns = [
            r'except\s*:',  # Bare except
            r'except\s+Exception\s*:',  # Too broad
            r'except.*:\s*pass',  # Silent failure
        ]
        
        for py_file in self.src_dir.rglob("*.py"):
            content = py_file.read_text()
            for pattern in bad_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    self.warnings.append({
                        'type': 'POOR_ERROR_HANDLING',
                        'file': str(py_file.relative_to(self.project_root)),
                        'line': line_num,
                        'severity': 'MEDIUM',
                        'message': 'Poor exception handling pattern'
                    })
        
        self._print_check_result("Exception Handling", 'POOR_ERROR_HANDLING', check_warnings=True)
    
    def check_resource_cleanup(self) -> None:
        """Check for resource cleanup issues."""
        print(f"Checking resource cleanup...")
        
        for py_file in self.src_dir.rglob("*.py"):
            content = py_file.read_text()
            
            # Check for open() without context manager
            if re.search(r'=\s*open\([^)]+\)', content):
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if re.search(r'=\s*open\([^)]+\)', line) and 'with' not in line:
                        self.warnings.append({
                            'type': 'RESOURCE_LEAK',
                            'file': str(py_file.relative_to(self.project_root)),
                            'line': i,
                            'severity': 'MEDIUM',
                            'message': 'File opened without context manager'
                        })
        
        self._print_check_result("Resource Cleanup", 'RESOURCE_LEAK', check_warnings=True)
    
    def _is_pickle_safe(self, file_path: Path, line_num: int) -> bool:
        """Check if pickle usage includes validation."""
        content = file_path.read_text()
        lines = content.split('\n')
        
        # Check surrounding lines for validation
        start = max(0, line_num - 5)
        end = min(len(lines), line_num + 5)
        
        context = '\n'.join(lines[start:end])
        
        # Look for signs of validation
        if any(check in context for check in 
               ['hmac', 'signature', 'verify', 'validate', 'trusted']):
            return True
        
        return False
    
    def _print_check_result(self, check_name: str, issue_type: str, check_warnings: bool = False) -> None:
        """Print result of a check."""
        if check_warnings:
            count = len([w for w in self.warnings if w['type'] == issue_type])
            if count > 0:
                print(f"  {YELLOW}⚠ Found {count} warnings{RESET}")
            else:
                print(f"  {GREEN}✓ No issues found{RESET}")
        else:
            count = len([i for i in self.issues if i['type'] == issue_type])
            if count > 0:
                print(f"  {RED}✗ Found {count} vulnerabilities{RESET}")
            else:
                print(f"  {GREEN}✓ Secure{RESET}")


def run_external_tools() -> Dict[str, bool]:
    """Run external security tools if available."""
    results = {}
    
    print(f"\n{BLUE}Running External Security Tools...{RESET}\n")
    
    # Bandit
    try:
        result = subprocess.run(
            ['bandit', '-r', 'src/', '-f', 'json'],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print(f"  {GREEN}✓ Bandit: No issues{RESET}")
            results['bandit'] = True
        else:
            print(f"  {YELLOW}⚠ Bandit: Found issues{RESET}")
            results['bandit'] = False
    except (subprocess.SubprocessError, FileNotFoundError):
        print(f"  {YELLOW}⚠ Bandit not installed{RESET}")
    
    # Safety
    try:
        result = subprocess.run(
            ['safety', 'check', '--json'],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print(f"  {GREEN}✓ Safety: No vulnerable dependencies{RESET}")
            results['safety'] = True
        else:
            print(f"  {RED}✗ Safety: Vulnerable dependencies found{RESET}")
            results['safety'] = False
    except (subprocess.SubprocessError, FileNotFoundError):
        print(f"  {YELLOW}⚠ Safety not installed{RESET}")
    
    return results


def main():
    """Main security check function."""
    print(f"\n{BLUE}═══════════════════════════════════════════{RESET}")
    print(f"{BLUE}   SECURITY AUDIT - Audio Extraction Project{RESET}")
    print(f"{BLUE}═══════════════════════════════════════════{RESET}\n")
    
    project_root = Path(__file__).parent.parent
    checker = SecurityChecker(project_root)
    
    # Run security checks
    all_passed, issues, warnings = checker.run_all_checks()
    
    # Run external tools
    external_results = run_external_tools()
    
    # Print summary
    print(f"\n{BLUE}═══════════════════════════════════════════{RESET}")
    print(f"{BLUE}                   SUMMARY{RESET}")
    print(f"{BLUE}═══════════════════════════════════════════{RESET}\n")
    
    if issues:
        print(f"{RED}CRITICAL ISSUES FOUND: {len(issues)}{RESET}\n")
        for issue in issues:
            print(f"  {RED}✗ [{issue['severity']}] {issue['type']}{RESET}")
            print(f"    File: {issue['file']}")
            if 'line' in issue:
                print(f"    Line: {issue['line']}")
            print(f"    {issue['message']}\n")
    
    if warnings:
        print(f"{YELLOW}WARNINGS: {len(warnings)}{RESET}\n")
        for warning in warnings[:5]:  # Show first 5 warnings
            print(f"  {YELLOW}⚠ [{warning['severity']}] {warning['type']}{RESET}")
            print(f"    File: {warning['file']}")
            if 'line' in warning:
                print(f"    Line: {warning['line']}")
            print(f"    {warning['message']}\n")
        
        if len(warnings) > 5:
            print(f"  {YELLOW}... and {len(warnings) - 5} more warnings{RESET}\n")
    
    # Final verdict
    print(f"{BLUE}═══════════════════════════════════════════{RESET}")
    if all_passed and all(external_results.values()):
        print(f"{GREEN}    ✅ SECURITY AUDIT PASSED{RESET}")
        print(f"{GREEN}    Ready for production deployment{RESET}")
        sys.exit(0)
    else:
        print(f"{RED}    ❌ SECURITY AUDIT FAILED{RESET}")
        print(f"{RED}    DO NOT DEPLOY TO PRODUCTION{RESET}")
        print(f"\n{YELLOW}Fix all critical issues before deployment.{RESET}")
        print(f"{YELLOW}See PRODUCTION_READINESS_REPORT.md for details.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()