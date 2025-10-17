#!/usr/bin/env python3
"""
Coverage summary script for synth-ai unit tests.
Generates a detailed coverage report and analysis.
"""

import subprocess
import sys
import os
import re
import argparse
from pathlib import Path


def run_coverage():
    """Run unit tests with coverage and generate reports."""
    print("🧪 Running unit tests with coverage...")
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Run pytest with coverage
    cmd = [
        "python", "-m", "pytest",
        "--cov=synth_ai",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--cov-report=xml",
        "--ignore=synth_ai/v0/",
        "tests/unit/",
        "-v"
    ]
    
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Coverage analysis completed successfully!")
        else:
            print("⚠️  Coverage analysis completed with test failures")
            print("Coverage data still generated despite test failures")
        return True
    except Exception as e:
        print(f"❌ Coverage analysis failed: {e}")
        return False


def parse_coverage_xml():
    """Parse coverage.xml to extract detailed metrics."""
    try:
        import xml.etree.ElementTree as ET
        
        if not Path("coverage.xml").exists():
            print("⚠️  coverage.xml not found. Run coverage first.")
            return None
            
        tree = ET.parse("coverage.xml")
        root = tree.getroot()
        
        # Extract overall metrics
        line_rate = float(root.get('line-rate', 0)) * 100
        branch_rate = float(root.get('branch-rate', 0)) * 100
        
        # Extract package-level metrics
        packages = []
        for package in root.findall('.//package'):
            name = package.get('name', 'unknown')
            pkg_line_rate = float(package.get('line-rate', 0)) * 100
            pkg_branch_rate = float(package.get('branch-rate', 0)) * 100
            
            # Count classes and methods
            classes = len(package.findall('.//class'))
            methods = len(package.findall('.//method'))
            
            packages.append({
                'name': name,
                'line_rate': pkg_line_rate,
                'branch_rate': pkg_branch_rate,
                'classes': classes,
                'methods': methods
            })
        
        return {
            'overall_line_rate': line_rate,
            'overall_branch_rate': branch_rate,
            'packages': packages
        }
        
    except ImportError:
        print("⚠️  xml.etree.ElementTree not available")
        return None
    except Exception as e:
        print(f"⚠️  Error parsing coverage.xml: {e}")
        return None


def generate_summary(coverage_data):
    """Generate a comprehensive coverage summary."""
    if not coverage_data:
        return "❌ No coverage data available"
    
    summary = []
    summary.append("## 🧪 Synth-AI Unit Test Coverage Report")
    summary.append("")
    summary.append(f"**Overall Coverage: {coverage_data['overall_line_rate']:.2f}%**")
    summary.append(f"**Branch Coverage: {coverage_data['overall_branch_rate']:.2f}%**")
    summary.append("")
    
    # Sort packages by coverage
    sorted_packages = sorted(coverage_data['packages'], 
                           key=lambda x: x['line_rate'], reverse=True)
    
    summary.append("### 📊 Top Modules by Coverage")
    for pkg in sorted_packages[:10]:  # Top 10
        if pkg['line_rate'] > 0:
            summary.append(f"- `{pkg['name']}`: {pkg['line_rate']:.2f}% "
                          f"({pkg['classes']} classes, {pkg['methods']} methods)")
    
    summary.append("")
    summary.append("### 🎯 Coverage Analysis")
    
    # Categorize packages
    high_coverage = [p for p in sorted_packages if p['line_rate'] >= 80]
    medium_coverage = [p for p in sorted_packages if 50 <= p['line_rate'] < 80]
    low_coverage = [p for p in sorted_packages if 0 < p['line_rate'] < 50]
    no_coverage = [p for p in sorted_packages if p['line_rate'] == 0]
    
    summary.append(f"- **High Coverage (≥80%)**: {len(high_coverage)} modules")
    summary.append(f"- **Medium Coverage (50-79%)**: {len(medium_coverage)} modules")
    summary.append(f"- **Low Coverage (1-49%)**: {len(low_coverage)} modules")
    summary.append(f"- **No Coverage (0%)**: {len(no_coverage)} modules")
    
    summary.append("")
    summary.append("### 📈 Recommendations")
    summary.append("1. **Focus on core modules**: Prioritize testing core API functionality")
    summary.append("2. **CLI testing**: Add comprehensive tests for command-line interfaces")
    summary.append("3. **Integration tests**: Implement tests for tracing v3 features")
    summary.append("4. **Learning algorithms**: Add tests for machine learning components")
    
    summary.append("")
    summary.append("### 🔍 Next Steps")
    summary.append("- Review the HTML coverage report for detailed line-by-line analysis")
    summary.append("- Identify critical paths with low coverage")
    summary.append("- Implement targeted tests for high-impact modules")
    
    summary.append("")
    summary.append("---")
    summary.append("*Generated by Blacksmith CI Worker*")
    
    return "\n".join(summary)


def update_readme_badge(coverage_percent):
    """Update the coverage badge in README.md with the current coverage percentage."""
    readme_path = Path(__file__).parent.parent / "README.md"
    
    if not readme_path.exists():
        print("⚠️  README.md not found, skipping badge update")
        return
    
    # Read current README
    with open(readme_path, 'r') as f:
        content = f.read()
    
    # Define coverage badge pattern
    coverage_pattern = r'!\[Coverage\]\(https://img\.shields\.io/badge/coverage-\d+\.\d+%25-[a-z]+\)'
    
    # Determine badge color based on coverage
    if coverage_percent >= 80:
        color = "brightgreen"
    elif coverage_percent >= 60:
        color = "green"
    elif coverage_percent >= 40:
        color = "yellow"
    else:
        color = "red"
    
    # Create new badge
    new_badge = f"![Coverage](https://img.shields.io/badge/coverage-{coverage_percent:.2f}%25-{color})"
    
    # Replace the badge
    updated_content = re.sub(coverage_pattern, new_badge, content)
    
    # Write back to file
    with open(readme_path, 'w') as f:
        f.write(updated_content)
    
    print(f"📝 Updated README.md coverage badge to {coverage_percent:.2f}%")


def main():
    """Main function to run coverage analysis and generate summary."""
    parser = argparse.ArgumentParser(description="Generate coverage report and update README badge")
    parser.add_argument("--update-readme", action="store_true", 
                       help="Update the coverage badge in README.md")
    parser.add_argument("--no-readme", action="store_true", 
                       help="Skip updating README.md badge")
    
    args = parser.parse_args()
    
    print("🚀 Starting synth-ai coverage analysis...")
    
    # Run coverage
    if not run_coverage():
        sys.exit(1)
    
    # Parse results
    coverage_data = parse_coverage_xml()
    
    # Generate summary
    summary = generate_summary(coverage_data)
    print("\n" + "="*60)
    print(summary)
    print("="*60)
    
    # Save summary to file
    with open("coverage_summary.md", "w") as f:
        f.write(summary)
    
    # Update README badge (default behavior unless --no-readme is specified)
    if not args.no_readme:
        coverage_percent = coverage_data['overall_line_rate']
        update_readme_badge(coverage_percent)
    elif args.update_readme:
        coverage_percent = coverage_data['overall_line_rate']
        update_readme_badge(coverage_percent)
    
    print(f"\n📄 Coverage summary saved to: coverage_summary.md")
    print(f"📊 HTML report available at: htmlcov/index.html")
    print(f"📈 XML report available at: coverage.xml")


if __name__ == "__main__":
    main()
