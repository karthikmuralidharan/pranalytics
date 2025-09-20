# GitHub PR Velocity Analytics Tool

A comprehensive command-line interface tool for analyzing GitHub pull request collaboration metrics and development velocity. This tool provides detailed insights into development team performance through statistical analysis of PR lifecycle events.

## Features

- **Velocity Metrics Calculation**: Automated calculation of critical development metrics
- **Tag-based Filtering**: Filter PRs by labels with flexible matching criteria
- **Statistical Analysis**: P50, P75, P90, P99 percentiles with trend analysis
- **CSV Export**: Structured data export with automatic date-prefixed filenames
- **SQLite Caching**: Performance optimization for large repositories
- **Enhanced Error Handling**: Robust retry logic and rate limiting
- **Author Grouping**: Optional grouping of metrics by PR author

## Installation

### Prerequisites

- Python 3.8 or higher
- UV package manager (recommended) or pip
- GitHub personal access token with repository read permissions

### Quick Setup with UV (Recommended)

1. Install UV if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone or download the project:
```bash
git clone https://github.com/your-repo/pr-velocity-analytics.git
cd pr-velocity-analytics
```

3. Install dependencies:
```bash
uv sync
```

4. Set up GitHub authentication:
```bash
export GITHUB_TOKEN="your_github_token_here"
```

### Alternative Setup with pip

1. Clone or download the script:
```bash
curl -O https://raw.githubusercontent.com/your-repo/pr_velocity_analytics.py
```

2. Install dependencies:
```bash
pip install PyGithub requests numpy python-dateutil
```

3. Set up GitHub authentication:
```bash
export GITHUB_TOKEN="your_github_token_here"
```

## Usage

### Basic Usage

With UV (recommended):
```bash
uv run python pr_velocity_analytics.py owner/repo --tags feature bug
```

Or as an executable script:
```bash
./pr_velocity_analytics.py owner/repo --tags feature bug
```

Traditional Python:
```bash
python pr_velocity_analytics.py owner/repo --tags feature bug
```

### Advanced Usage

Filter by date and group by author:
```bash
uv run python pr_velocity_analytics.py owner/repo \
    --tags feature enhancement \
    --since 30d \
    --group-by-author \
    --output custom_report.csv
```

### Command Line Arguments

| Argument | Description | Required | Example |
|----------|-------------|----------|---------|
| `repo` | GitHub repository in format `owner/repo` | Yes | `microsoft/vscode` |
| `--tags` | Filter PRs by labels (space-separated) | Yes | `--tags feature bug enhancement` |
| `--since` | Filter PRs created after date | No | `--since 30d` or `--since 2024-01-15` |
| `--group-by-author` | Group metrics by PR author | No | `--group-by-author` |
| `--output` | Output CSV filename | No | `--output my_report.csv` |

### Date Format Support

The `--since` parameter supports multiple date formats:

- **Relative dates**: `30d`, `7d`, `90d` (days ago)
- **ISO format**: `2024-01-15`, `2024-12-31`
- **Alternative formats**: `2024/01/15`, `01/15/2024`, `15/01/2024`

## Velocity Metrics Explained

### 1. Cycle Time
**Definition**: Total time from PR creation to merge completion.

**Calculation**: `merged_at - created_at`

**Business Value**: Measures overall development velocity and identifies bottlenecks in the development process.

### 2. First Response Time
**Definition**: Time from PR creation to first review or comment.

**Calculation**: `min(first_review_time, first_comment_time) - created_at`

**Business Value**: Indicates team responsiveness and code review engagement levels.

### 3. Rework Time
**Definition**: Time spent addressing review feedback between changes requested and approval.

**Calculation**: `approved_at - changes_requested_at` (for each rework cycle)

**Business Value**: Measures code quality and identifies areas needing improvement in development practices.

### 4. Merge Time
**Definition**: Time from final approval to merge completion.

**Calculation**: `merged_at - final_approval_at`

**Business Value**: Identifies deployment pipeline efficiency and merge process bottlenecks.

## Statistical Analysis

### Percentile Calculations

The tool calculates key percentiles for each metric:

- **P50 (Median)**: 50% of PRs complete faster than this time
- **P75**: 75% of PRs complete faster than this time  
- **P90**: 90% of PRs complete faster than this time
- **P99**: 99% of PRs complete faster than this time

### Interpretation Guide

| Percentile | Interpretation | Use Case |
|------------|----------------|----------|
| P50 | Typical performance | Sprint planning and estimation |
| P75 | Good performance threshold | SLA target setting |
| P90 | Excellence threshold | Performance optimization goals |
| P99 | Outlier identification | Process improvement focus areas |

## Output Format

### CSV Export Structure

```csv
PR Number,Title,Author,Labels,Created At,Cycle Time (hours),First Response Time (hours),Rework Time (hours),Merge Time (hours)
123,"Add user authentication",developer1,"feature,security",2024-01-15T09:00:00Z,78.5,1.25,26.5,1.5
124,"Fix login bug",developer2,"bug,urgent",2024-01-16T14:00:00Z,24.0,0.5,8.0,0.25
...
```

### Statistical Summary

The CSV includes aggregated statistics at the end:

```csv
Metric,Count,Mean,Std Dev,P50 (Median),P75,P90,P99
Cycle Time,150,45.2,28.7,38.5,58.2,78.9,125.4
First Response Time,150,3.1,4.2,1.8,4.5,8.2,15.7
Rework Time,89,12.4,18.3,6.2,18.4,35.7,68.9
Merge Time,150,1.8,2.1,1.2,2.4,4.2,8.9
```

## Performance Optimization

### Caching Strategy

The tool uses SQLite-based caching to optimize performance:

- **PR Data Caching**: Basic PR information cached indefinitely
- **Review Caching**: Review data cached with automatic invalidation
- **Comment Caching**: Comment data cached for improved response times
- **Cache Location**: `./pr_cache.db` (automatically created)

### Rate Limiting

GitHub API rate limiting is handled automatically:

- **Retry Logic**: Exponential backoff with jitter
- **Rate Limit Detection**: Automatic detection and waiting
- **Error Recovery**: Graceful handling of temporary failures

## Examples

### Example 1: Sprint Retrospective Analysis

Analyze development velocity for the last sprint:

```bash
python pr_velocity_analytics.py myorg/myproject \
    --tags feature bug \
    --since 14d \
    --output sprint_retrospective.csv
```

**Use Case**: Identify bottlenecks and improvement opportunities for next sprint planning.

### Example 2: Team Performance Comparison

Compare performance across team members:

```bash
python pr_velocity_analytics.py myorg/myproject \
    --tags feature \
    --since 90d \
    --group-by-author \
    --output team_performance.csv
```

**Use Case**: Identify mentoring opportunities and workload distribution insights.

### Example 3: Bug Fix Efficiency Analysis

Analyze bug fix turnaround times:

```bash
python pr_velocity_analytics.py myorg/myproject \
    --tags bug critical urgent \
    --since 180d \
    --output bug_analysis.csv
```

**Use Case**: Establish bug fix SLAs and identify process improvements.

### Example 4: Release Readiness Assessment

Evaluate pre-release development velocity:

```bash
python pr_velocity_analytics.py myorg/myproject \
    --tags release-prep feature \
    --since 2024-10-01 \
    --output release_velocity.csv
```

**Use Case**: Assess release timeline feasibility and resource allocation.

## Troubleshooting

### Common Issues

#### Authentication Errors
```
Error: GitHub authentication failed
```
**Solution**: Verify your GitHub token has repository access permissions.

#### Rate Limit Exceeded
```
Rate limit exceeded. Waiting 3600 seconds...
```
**Solution**: The tool automatically handles rate limits. For faster processing, consider using a GitHub App token.

#### No PRs Found
```
No PRs found matching the specified criteria
```
**Solution**: Verify tag names match exactly (case-sensitive) and check date ranges.

#### Cache Issues
```
Error: Cannot access cache database
```
**Solution**: Ensure write permissions in the current directory or delete `pr_cache.db`.

### Debug Mode

For detailed debugging, modify the script to enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Configuration

### Custom Cache Location

Modify the cache database location:

```python
cache = PRVelocityCache('/custom/path/pr_cache.db')
```

### GitHub Enterprise Support

For GitHub Enterprise instances:

```python
from github import Github
github_client = Github(base_url="https://github.enterprise.com/api/v3", login_or_token=token)
```

## Contributing

### Development Setup with UV

1. Clone the repository
2. Install development dependencies:
```bash
uv sync --dev
```

3. Run tests:
```bash
uv run pytest test_pr_velocity_analytics.py -v
```

4. Set up development environment:
```bash
make dev-setup
```

### Alternative Development Setup with pip

1. Clone the repository
2. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

3. Run tests:
```bash
python -m pytest test_pr_velocity_analytics.py -v
```

### Test Coverage

Run tests with coverage using UV:
```bash
uv run pytest test_pr_velocity_analytics.py --cov=pr_velocity_analytics --cov-report=html
```

Or with Make:
```bash
make test-coverage
```

### Code Quality

Format code:
```bash
make format
```

Run linting:
```bash
make lint
```

## License

MIT License - see LICENSE file for details.

## Support

For issues and feature requests, please create an issue in the project repository.

## Changelog

### Version 1.0.0
- Initial release with core velocity metrics
- SQLite caching implementation
- Statistical analysis with percentiles
- CSV export functionality
- Comprehensive error handling and retry logic
