#!/usr/bin/env python3
"""
GitHub PR Velocity Analytics Tool

A comprehensive CLI tool for analyzing GitHub pull request collaboration metrics
including cycle time, first response time, rework time, and merge time with
statistical analysis and CSV export capabilities.

Based on the existing github_report_generator.py foundation.
"""

import os
import sys
import json
import argparse
import sqlite3
import csv
import statistics
import random
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from functools import wraps
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    from github import Github, GithubException
    from github.PullRequest import PullRequest
    from github.Repository import Repository
    import numpy as np
except ImportError as e:
    # Use basic logging for dependency errors before logger is configured
    print("❌ Missing required dependencies. Please install:")
    print("   pip install PyGithub requests numpy")
    sys.exit(1)


# Configure logging
def setup_logging(verbose: bool = False):
    """Setup logging configuration with optional verbose mode"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter for different log levels
    class CustomFormatter(logging.Formatter):
        """Custom formatter with emojis and colors"""
        
        FORMATS = {
            logging.DEBUG: "🔍 [DEBUG] %(message)s",
            logging.INFO: "ℹ️  %(message)s",
            logging.WARNING: "⚠️  %(message)s",
            logging.ERROR: "❌ %(message)s",
            logging.CRITICAL: "💥 %(message)s"
        }
        
        def format(self, record):
            log_format = self.FORMATS.get(record.levelno, "%(message)s")
            formatter = logging.Formatter(log_format)
            return formatter.format(record)
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(CustomFormatter())
    
    logger.addHandler(console_handler)
    
    return logger


# Global logger instance
logger = logging.getLogger(__name__)


def retry_on_github_error(max_retries: int = 3, backoff_factor: float = 2.0):
    """Decorator for retrying GitHub API calls with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except GithubException as e:
                    last_exception = e
                    
                    # Don't retry on authentication or permission errors
                    if e.status in [401, 403, 404]:
                        raise e
                    
                    # For rate limit errors, wait for reset
                    if e.status == 403 and 'rate limit' in str(e.data).lower():
                        logger.warning("Rate limit exceeded. Waiting for reset...")
                        time.sleep(60)  # Wait 1 minute before retry
                        continue
                    
                    # For other server errors, use exponential backoff
                    if e.status >= 500:
                        wait_time = backoff_factor ** attempt + random.uniform(0, 1)
                        logger.warning(f"Server error (attempt {attempt + 1}/{max_retries + 1}). Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    
                    # For other errors, don't retry
                    raise e
                
                except (requests.exceptions.RequestException,
                       requests.exceptions.ConnectionError,
                       requests.exceptions.Timeout) as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        wait_time = backoff_factor ** attempt + random.uniform(0, 1)
                        logger.warning(f"Network error (attempt {attempt + 1}/{max_retries + 1}). Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    
                    raise e
                
                except Exception as e:
                    last_exception = e
                    
                    # For testing purposes, retry generic exceptions too
                    if attempt < max_retries:
                        wait_time = backoff_factor * (2 ** attempt)
                        print(f"🔄 Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    
                    raise e
            
            # If all retries failed, raise the last exception
            raise last_exception
        
        return wrapper
    return decorator


class GitHubAPIError(Exception):
    """Custom exception for GitHub API related errors"""
    pass


class DataProcessingError(Exception):
    """Custom exception for data processing related errors"""
    pass


@dataclass
class PRVelocityMetrics:
    """Data class to hold PR velocity metrics"""
    pr_number: int
    title: str
    author: str
    labels: List[str]
    created_at: datetime
    first_review_at: Optional[datetime] = None
    first_comment_at: Optional[datetime] = None
    changes_requested_at: Optional[datetime] = None
    approved_at: Optional[datetime] = None
    merged_at: Optional[datetime] = None
    cycle_time_hours: Optional[float] = None
    first_response_time_hours: Optional[float] = None
    rework_time_hours: Optional[float] = None
    merge_time_hours: Optional[float] = None
    # Enhanced communication delay metrics
    avg_reviewer_to_author_hours: Optional[float] = None
    avg_author_to_reviewer_hours: Optional[float] = None
    max_communication_gap_hours: Optional[float] = None
    communication_rounds_count: Optional[int] = None


@dataclass
class MetricSummary:
    """Statistical summary for velocity metrics"""
    metric_name: str
    count: int
    p50_median: float
    p75: float
    p90: float
    p99: float
    mean: float
    std_dev: float


@dataclass
class AnalyticsConfig:
    """Configuration for PR velocity analytics"""
    github_token: str
    repo_name: str
    tags: List[str]
    since_date: Optional[datetime] = None
    group_by_author: bool = False
    output_filename: Optional[str] = None


class PRVelocityCache:
    """SQLite-based caching for GitHub API responses"""
    
    def __init__(self, cache_file: str = "pr_velocity_cache.db"):
        self.cache_file = cache_file
        self._init_cache()
    
    def _init_cache(self):
        """Initialize SQLite cache database"""
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()
        
        # Create tables for caching
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pr_data (
                repo_name TEXT,
                pr_number INTEGER,
                data_json TEXT,
                cached_at TIMESTAMP,
                PRIMARY KEY (repo_name, pr_number)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pr_reviews (
                repo_name TEXT,
                pr_number INTEGER,
                reviews_json TEXT,
                cached_at TIMESTAMP,
                PRIMARY KEY (repo_name, pr_number)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pr_comments (
                repo_name TEXT,
                pr_number INTEGER,
                comments_json TEXT,
                cached_at TIMESTAMP,
                PRIMARY KEY (repo_name, pr_number)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pr_review_comments (
                repo_name TEXT,
                pr_number INTEGER,
                review_comments_json TEXT,
                cached_at TIMESTAMP,
                PRIMARY KEY (repo_name, pr_number)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_cached_pr_data(self, repo_name: str, pr_number: int) -> Optional[Dict]:
        """Retrieve cached PR data"""
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT data_json FROM pr_data 
            WHERE repo_name = ? AND pr_number = ?
        """, (repo_name, pr_number))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        return None
    
    def cache_pr_data(self, repo_name: str, pr_number: int, data: Dict):
        """Cache PR data"""
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO pr_data 
            (repo_name, pr_number, data_json, cached_at)
            VALUES (?, ?, ?, ?)
        """, (repo_name, pr_number, json.dumps(data), datetime.now()))
        
        conn.commit()
        conn.close()
    
    def get_cached_reviews(self, repo_name: str, pr_number: int) -> Optional[List]:
        """Retrieve cached PR reviews"""
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT reviews_json FROM pr_reviews 
            WHERE repo_name = ? AND pr_number = ?
        """, (repo_name, pr_number))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        return None
    
    def cache_reviews(self, repo_name: str, pr_number: int, reviews: List):
        """Cache PR reviews"""
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO pr_reviews 
            (repo_name, pr_number, reviews_json, cached_at)
            VALUES (?, ?, ?, ?)
        """, (repo_name, pr_number, json.dumps(reviews), datetime.now()))
        
        conn.commit()
        conn.close()
    
    def get_cached_comments(self, repo_name: str, pr_number: int) -> Optional[List]:
        """Retrieve cached PR comments"""
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT comments_json FROM pr_comments 
            WHERE repo_name = ? AND pr_number = ?
        """, (repo_name, pr_number))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        return None
    
    def cache_comments(self, repo_name: str, pr_number: int, comments: List):
        """Cache PR comments"""
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO pr_comments
            (repo_name, pr_number, comments_json, cached_at)
            VALUES (?, ?, ?, ?)
        """, (repo_name, pr_number, json.dumps(comments), datetime.now()))
        
        conn.commit()
        conn.close()
    
    def get_cached_review_comments(self, repo_name: str, pr_number: int) -> Optional[List]:
        """Retrieve cached PR review comments"""
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT review_comments_json FROM pr_review_comments
            WHERE repo_name = ? AND pr_number = ?
        """, (repo_name, pr_number))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        return None
    
    def cache_review_comments(self, repo_name: str, pr_number: int, review_comments: List):
        """Cache PR review comments"""
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO pr_review_comments
            (repo_name, pr_number, review_comments_json, cached_at)
            VALUES (?, ?, ?, ?)
        """, (repo_name, pr_number, json.dumps(review_comments), datetime.now()))
        
        conn.commit()
        conn.close()


class EnhancedGitHubAPIClient:
    """Enhanced GitHub API client with caching and velocity-specific data collection"""
    
    def __init__(self, token: str):
        """Initialize GitHub client with authentication and caching"""
        if not token:
            raise ValueError("GitHub token is required")
        
        self.github = Github(token, per_page=100)
        self.cache = PRVelocityCache()
        self.rate_limit_remaining = None
        self._validate_authentication()
    
    def _validate_authentication(self):
        """Validate GitHub token and check API access"""
        try:
            logger.debug("Authenticating with GitHub API...")
            user = self.github.get_user()
            logger.info(f"Authenticated as: {user.login}")
            rate_limit = self.github.get_rate_limit()
            self.rate_limit_remaining = rate_limit.core.remaining
            logger.info(f"API Rate Limit: {self.rate_limit_remaining}/{rate_limit.core.limit}")
            logger.debug("GitHub API client initialized successfully")
        except GithubException as e:
            if e.status == 401:
                raise ValueError("Invalid GitHub token. Please check your authentication.")
            elif e.status == 403:
                raise ValueError("GitHub token doesn't have sufficient permissions.")
            else:
                raise ValueError(f"GitHub API error: {e.data.get('message', 'Unknown error')}")
    
    def _handle_rate_limit(self):
        """Handle GitHub API rate limiting with exponential backoff"""
        rate_limit = self.github.get_rate_limit()
        if rate_limit.core.remaining < 10:
            reset_time = rate_limit.core.reset
            wait_time = (reset_time - datetime.now(timezone.utc)).total_seconds() + 60
            if wait_time > 0:
                logger.warning(f"Rate limit approaching. Waiting {wait_time:.0f} seconds...")
                time.sleep(wait_time)
    
    def get_repository(self, repo_name: str) -> Repository:
        """Get repository object with error handling"""
        try:
            return self.github.get_repo(repo_name)
        except GithubException as e:
            if e.status == 404:
                raise ValueError(f"Repository '{repo_name}' not found or not accessible.")
            else:
                raise ValueError(f"Error accessing repository: {e.data.get('message', 'Unknown error')}")
    
    def collect_pr_timeline_data(self, repo: Repository, pr: PullRequest) -> Dict:
        """Collect comprehensive timeline data for a PR"""
        repo_name = repo.full_name
        pr_number = pr.number
        
        # Check cache first
        cached_data = self.cache.get_cached_pr_data(repo_name, pr_number)
        if cached_data:
            return cached_data
        
        self._handle_rate_limit()
        
        timeline_data = {
            'pr_number': pr_number,
            'title': pr.title,
            'author': pr.user.login,
            'labels': [label.name for label in pr.labels],
            'created_at': pr.created_at.isoformat(),
            'merged_at': pr.merged_at.isoformat() if pr.merged_at else None,
            'state': pr.state,
            'reviews': [],
            'comments': [],
            'review_comments': []
        }
        
        # Collect reviews
        try:
            cached_reviews = self.cache.get_cached_reviews(repo_name, pr_number)
            if cached_reviews:
                timeline_data['reviews'] = cached_reviews
            else:
                reviews_data = []
                reviews = pr.get_reviews()
                for review in reviews:
                    reviews_data.append({
                        'submitted_at': review.submitted_at.isoformat() if review.submitted_at else None,
                        'state': review.state,
                        'user': review.user.login
                    })
                timeline_data['reviews'] = reviews_data
                self.cache.cache_reviews(repo_name, pr_number, reviews_data)
        except GithubException as e:
            logger.warning(f"Could not fetch reviews for PR #{pr_number}: {e.data.get('message', 'Unknown error')}")
        
        # Collect comments
        try:
            cached_comments = self.cache.get_cached_comments(repo_name, pr_number)
            if cached_comments:
                timeline_data['comments'] = cached_comments
            else:
                comments_data = []
                comments = pr.get_issue_comments()
                for comment in comments:
                    comments_data.append({
                        'created_at': comment.created_at.isoformat(),
                        'user': comment.user.login
                    })
                timeline_data['comments'] = comments_data
                self.cache.cache_comments(repo_name, pr_number, comments_data)
        except GithubException as e:
            logger.warning(f"Could not fetch comments for PR #{pr_number}: {e.data.get('message', 'Unknown error')}")
        
        # Collect review comments (inline code review comments)
        try:
            cached_review_comments = self.cache.get_cached_review_comments(repo_name, pr_number)
            if cached_review_comments:
                timeline_data['review_comments'] = cached_review_comments
            else:
                review_comments_data = []
                review_comments = pr.get_review_comments()
                for review_comment in review_comments:
                    review_comments_data.append({
                        'created_at': review_comment.created_at.isoformat(),
                        'user': review_comment.user.login,
                        'in_reply_to_id': review_comment.in_reply_to_id
                    })
                timeline_data['review_comments'] = review_comments_data
                self.cache.cache_review_comments(repo_name, pr_number, review_comments_data)
        except GithubException as e:
            logger.warning(f"Could not fetch review comments for PR #{pr_number}: {e.data.get('message', 'Unknown error')}")
        
        # Cache the complete timeline data
        self.cache.cache_pr_data(repo_name, pr_number, timeline_data)
        
        return timeline_data
    
    def collect_pr_timeline_data_concurrent(self, repo: Repository, pr: PullRequest) -> Dict:
        """Collect comprehensive timeline data for a PR using concurrent API calls"""
        repo_name = repo.full_name
        pr_number = pr.number
        
        # Check cache first
        cached_data = self.cache.get_cached_pr_data(repo_name, pr_number)
        if cached_data:
            return cached_data
        
        self._handle_rate_limit()
        
        timeline_data = {
            'pr_number': pr_number,
            'title': pr.title,
            'author': pr.user.login,
            'labels': [label.name for label in pr.labels],
            'created_at': pr.created_at.isoformat(),
            'merged_at': pr.merged_at.isoformat() if pr.merged_at else None,
            'state': pr.state,
            'reviews': [],
            'comments': [],
            'review_comments': []
        }
        
        # Use ThreadPoolExecutor for concurrent API calls
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit concurrent tasks for reviews, comments, and review comments
            future_to_data_type = {}
            
            # Reviews
            cached_reviews = self.cache.get_cached_reviews(repo_name, pr_number)
            if cached_reviews:
                timeline_data['reviews'] = cached_reviews
            else:
                future_to_data_type[executor.submit(self._fetch_reviews, pr)] = 'reviews'
            
            # Comments
            cached_comments = self.cache.get_cached_comments(repo_name, pr_number)
            if cached_comments:
                timeline_data['comments'] = cached_comments
            else:
                future_to_data_type[executor.submit(self._fetch_comments, pr)] = 'comments'
            
            # Review comments
            cached_review_comments = self.cache.get_cached_review_comments(repo_name, pr_number)
            if cached_review_comments:
                timeline_data['review_comments'] = cached_review_comments
            else:
                future_to_data_type[executor.submit(self._fetch_review_comments, pr)] = 'review_comments'
            
            # Collect results as they complete
            for future in as_completed(future_to_data_type):
                data_type = future_to_data_type[future]
                try:
                    result = future.result()
                    timeline_data[data_type] = result
                    
                    # Cache the results
                    if data_type == 'reviews':
                        self.cache.cache_reviews(repo_name, pr_number, result)
                    elif data_type == 'comments':
                        self.cache.cache_comments(repo_name, pr_number, result)
                    elif data_type == 'review_comments':
                        self.cache.cache_review_comments(repo_name, pr_number, result)
                        
                except Exception as e:
                    logger.warning(f"Could not fetch {data_type} for PR #{pr_number}: {e}")
                    timeline_data[data_type] = []
        
        # Cache the complete timeline data
        self.cache.cache_pr_data(repo_name, pr_number, timeline_data)
        
        return timeline_data
    
    def _fetch_reviews(self, pr: PullRequest) -> List[Dict]:
        """Fetch reviews for a PR"""
        try:
            reviews_data = []
            reviews = pr.get_reviews()
            for review in reviews:
                reviews_data.append({
                    'submitted_at': review.submitted_at.isoformat() if review.submitted_at else None,
                    'state': review.state,
                    'user': review.user.login
                })
            return reviews_data
        except GithubException as e:
            logger.warning(f"Could not fetch reviews: {e.data.get('message', 'Unknown error')}")
            return []
    
    def _fetch_comments(self, pr: PullRequest) -> List[Dict]:
        """Fetch comments for a PR"""
        try:
            comments_data = []
            comments = pr.get_issue_comments()
            for comment in comments:
                comments_data.append({
                    'created_at': comment.created_at.isoformat(),
                    'user': comment.user.login
                })
            return comments_data
        except GithubException as e:
            logger.warning(f"Could not fetch comments: {e.data.get('message', 'Unknown error')}")
            return []
    
    def _fetch_review_comments(self, pr: PullRequest) -> List[Dict]:
        """Fetch review comments for a PR"""
        try:
            review_comments_data = []
            review_comments = pr.get_review_comments()
            for review_comment in review_comments:
                review_comments_data.append({
                    'created_at': review_comment.created_at.isoformat(),
                    'user': review_comment.user.login,
                    'in_reply_to_id': review_comment.in_reply_to_id
                })
            return review_comments_data
        except GithubException as e:
            logger.warning(f"Could not fetch review comments: {e.data.get('message', 'Unknown error')}")
            return []
    
    def process_prs_concurrently(self, repo: Repository, prs: List[PullRequest], max_workers: int = 10) -> List[Optional[Dict]]:
        """Process multiple PRs concurrently with batch processing"""
        logger.info(f"🚀 Processing {len(prs)} PRs with {max_workers} concurrent workers...")
        
        results = []
        processed_count = 0
        error_count = 0
        
        # Process PRs in batches to manage memory and API rate limits
        batch_size = max_workers * 2  # Process in batches of 2x worker count
        
        for i in range(0, len(prs), batch_size):
            batch = prs[i:i + batch_size]
            logger.debug(f"📦 Processing batch {i//batch_size + 1}/{(len(prs) + batch_size - 1)//batch_size} ({len(batch)} PRs)")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all PRs in the current batch
                future_to_pr = {
                    executor.submit(self._process_single_pr_safe, repo, pr): pr
                    for pr in batch
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_pr):
                    pr = future_to_pr[future]
                    try:
                        result = future.result()
                        results.append(result)
                        if result is not None:
                            processed_count += 1
                        else:
                            error_count += 1
                    except Exception as e:
                        logger.warning(f"❌ Error processing PR #{pr.number}: {e}")
                        results.append(None)
                        error_count += 1
                    
                    # Progress reporting
                    total_processed = processed_count + error_count
                    if total_processed % 25 == 0:
                        logger.info(f"📊 Progress: {total_processed}/{len(prs)} PRs processed ({processed_count} success, {error_count} errors)")
            
            # Small delay between batches to be respectful to GitHub API
            if i + batch_size < len(prs):
                time.sleep(0.5)
        
        logger.info(f"✅ Batch processing complete: {processed_count} successful, {error_count} errors")
        return results
    
    def _process_single_pr_safe(self, repo: Repository, pr: PullRequest) -> Optional[Dict]:
        """Safely process a single PR with error handling"""
        try:
            timeline_data = self.collect_pr_timeline_data_concurrent(repo, pr)
            metrics = VelocityMetricsCalculator.calculate_metrics(timeline_data)
            return metrics
        except Exception as e:
            logger.debug(f"🔍 Error processing PR #{pr.number}: {e}")
            return None


class VelocityMetricsCalculator:
    """Calculate velocity metrics from PR timeline data"""
    
    @staticmethod
    def calculate_metrics(timeline_data: Dict) -> PRVelocityMetrics:
        """Calculate all velocity metrics for a PR including communication delay metrics"""
        # Parse timeline data
        pr_number = timeline_data['pr_number']
        title = timeline_data['title']
        author = timeline_data['author']
        labels = timeline_data['labels']

        try:
            created_at = datetime.fromisoformat(timeline_data['created_at'].replace('Z', '+00:00'))
        except ValueError:
            # Handle malformed dates gracefully
            created_at = datetime.now()
        merged_at = None
        if timeline_data['merged_at']:
            merged_at = datetime.fromisoformat(timeline_data['merged_at'].replace('Z', '+00:00'))
        
        # Initialize metrics object
        metrics = PRVelocityMetrics(
            pr_number=pr_number,
            title=title,
            author=author,
            labels=labels,
            created_at=created_at,
            merged_at=merged_at
        )
        
        # Find first review and first comment timestamps
        first_review_at = None
        changes_requested_at = None
        approved_at = None
        
        for review in timeline_data['reviews']:
            if review['submitted_at']:
                review_time = datetime.fromisoformat(review['submitted_at'].replace('Z', '+00:00'))
                
                # Track first review
                if first_review_at is None or review_time < first_review_at:
                    first_review_at = review_time
                
                # Track changes requested
                if review['state'] == 'CHANGES_REQUESTED':
                    if changes_requested_at is None or review_time > changes_requested_at:
                        changes_requested_at = review_time
                
                # Track final approval
                if review['state'] == 'APPROVED':
                    if approved_at is None or review_time > approved_at:
                        approved_at = review_time
        
        first_comment_at = None
        for comment in timeline_data['comments']:
            comment_time = datetime.fromisoformat(comment['created_at'].replace('Z', '+00:00'))
            if first_comment_at is None or comment_time < first_comment_at:
                first_comment_at = comment_time
        
        # Set timeline data
        metrics.first_review_at = first_review_at
        metrics.first_comment_at = first_comment_at
        metrics.changes_requested_at = changes_requested_at
        metrics.approved_at = approved_at
        
        # Calculate traditional velocity metrics
        
        # 1. Cycle Time: PR creation to merge completion
        if merged_at:
            metrics.cycle_time_hours = (merged_at - created_at).total_seconds() / 3600
        
        # 2. First Response Time: PR creation to first review or comment
        first_response_at = None
        if first_review_at and first_comment_at:
            first_response_at = min(first_review_at, first_comment_at)
        elif first_review_at:
            first_response_at = first_review_at
        elif first_comment_at:
            first_response_at = first_comment_at
        
        if first_response_at:
            metrics.first_response_time_hours = (first_response_at - created_at).total_seconds() / 3600
        
        # 3. Rework Time: Changes requested to final approval or merge
        if changes_requested_at:
            rework_end = approved_at or merged_at
            if rework_end and rework_end > changes_requested_at:
                metrics.rework_time_hours = (rework_end - changes_requested_at).total_seconds() / 3600
        
        # 4. Merge Time: Final approval to merge completion
        if approved_at and merged_at and merged_at > approved_at:
            metrics.merge_time_hours = (merged_at - approved_at).total_seconds() / 3600
        
        # Calculate enhanced communication delay metrics
        communication_metrics = VelocityMetricsCalculator._calculate_communication_metrics(
            timeline_data, author, created_at
        )
        
        metrics.avg_reviewer_to_author_hours = communication_metrics['avg_reviewer_to_author_hours']
        metrics.avg_author_to_reviewer_hours = communication_metrics['avg_author_to_reviewer_hours']
        metrics.max_communication_gap_hours = communication_metrics['max_communication_gap_hours']
        metrics.communication_rounds_count = communication_metrics['communication_rounds_count']
        
        return metrics
    
    @staticmethod
    def _calculate_communication_metrics(timeline_data: Dict, author: str, created_at: datetime) -> Dict:
        """Calculate communication delay metrics from timeline data"""
        # Collect all interactions with timestamps and participant info
        all_interactions = []
        
        # Add reviews
        for review in timeline_data['reviews']:
            if review['submitted_at'] and review['user']:
                interaction_time = datetime.fromisoformat(review['submitted_at'].replace('Z', '+00:00'))
                all_interactions.append({
                    'timestamp': interaction_time,
                    'user': review['user'],
                    'type': 'review',
                    'is_author': review['user'] == author
                })
        
        # Add comments
        for comment in timeline_data['comments']:
            if comment['created_at'] and comment['user']:
                interaction_time = datetime.fromisoformat(comment['created_at'].replace('Z', '+00:00'))
                all_interactions.append({
                    'timestamp': interaction_time,
                    'user': comment['user'],
                    'type': 'comment',
                    'is_author': comment['user'] == author
                })
        
        # Add review comments (inline code review comments)
        for review_comment in timeline_data.get('review_comments', []):
            if review_comment['created_at'] and review_comment['user']:
                interaction_time = datetime.fromisoformat(review_comment['created_at'].replace('Z', '+00:00'))
                all_interactions.append({
                    'timestamp': interaction_time,
                    'user': review_comment['user'],
                    'type': 'review_comment',
                    'is_author': review_comment['user'] == author
                })
        
        # Sort interactions by timestamp
        all_interactions.sort(key=lambda x: x['timestamp'])
        
        if len(all_interactions) < 2:
            return {
                'avg_reviewer_to_author_hours': None,
                'avg_author_to_reviewer_hours': None,
                'max_communication_gap_hours': None,
                'communication_rounds_count': None
            }
        
        # Calculate reviewer-to-author response times
        reviewer_to_author_times = []
        author_to_reviewer_times = []
        communication_gaps = []
        
        # Find response patterns
        for i in range(1, len(all_interactions)):
            current = all_interactions[i]
            previous = all_interactions[i-1]
            
            time_diff_hours = (current['timestamp'] - previous['timestamp']).total_seconds() / 3600
            communication_gaps.append(time_diff_hours)
            
            # Reviewer responding to author
            if not current['is_author'] and previous['is_author']:
                reviewer_to_author_times.append(time_diff_hours)
            
            # Author responding to reviewer
            elif current['is_author'] and not previous['is_author']:
                author_to_reviewer_times.append(time_diff_hours)
        
        # Calculate communication rounds (back-and-forth cycles)
        rounds = 0
        last_participant_type = None
        
        for interaction in all_interactions:
            current_participant_type = 'author' if interaction['is_author'] else 'reviewer'
            if last_participant_type and last_participant_type != current_participant_type:
                rounds += 1
            last_participant_type = current_participant_type
        
        # Convert to round-trip cycles
        communication_rounds_count = rounds // 2 if rounds > 0 else 0
        
        return {
            'avg_reviewer_to_author_hours': statistics.mean(reviewer_to_author_times) if reviewer_to_author_times else None,
            'avg_author_to_reviewer_hours': statistics.mean(author_to_reviewer_times) if author_to_reviewer_times else None,
            'max_communication_gap_hours': max(communication_gaps) if communication_gaps else None,
            'communication_rounds_count': communication_rounds_count if communication_rounds_count > 0 else None
        }


class StatisticalAnalyzer:
    """Statistical analysis for velocity metrics"""

    @staticmethod
    def calculate_percentiles(values: List[float]) -> Dict[str, float]:
        """Calculate P50, P75, P90, P99 percentiles"""
        if not values:
            return {'p50': 0, 'p75': 0, 'p90': 0, 'p99': 0}

        return {
            'p50': np.percentile(values, 50),
            'p75': np.percentile(values, 75),
            'p90': np.percentile(values, 90),
            'p99': np.percentile(values, 99)
        }

    @staticmethod
    def generate_metric_summary(metrics_list: List[PRVelocityMetrics], metric_name: str) -> MetricSummary:
        """Generate statistical summary for a specific metric"""
        # Map legacy metric names to actual attribute names
        metric_mapping = {
            'cycle_time': 'cycle_time_hours',
            'rework_time': 'rework_time_hours',
            'first_response_time': 'first_response_time_hours',
            'merge_time': 'merge_time_hours'
        }
        
        actual_metric_name = metric_mapping.get(metric_name, metric_name)
        
        # Extract non-null values for the metric
        values = []
        for metric in metrics_list:
            value = getattr(metric, actual_metric_name, None)
            if value is not None:
                values.append(value)

        if not values:
            return MetricSummary(
                metric_name=metric_name.replace('_', ' ').title(),
                count=0,
                p50_median=0,
                p75=0,
                p90=0,
                p99=0,
                mean=0,
                std_dev=0
            )

        percentiles = StatisticalAnalyzer.calculate_percentiles(values)

        return MetricSummary(
            metric_name=metric_name.replace('_', ' ').title(),
            count=len(values),
            p50_median=percentiles['p50'] if values else 0,
            p75=percentiles['p75'] if values else 0,
            p90=percentiles['p90'] if values else 0,
            p99=percentiles['p99'] if values else 0,
            mean=statistics.mean(values) if values else 0,
            std_dev=statistics.stdev(values) if len(values) > 1 else 0
        )


def parse_date(date_string: str) -> datetime:
    """Parse date string in various formats"""
    # Handle relative dates like "30d", "7d", etc.
    if date_string.endswith('d'):
        try:
            days = int(date_string[:-1])
            return datetime.now(timezone.utc) - timedelta(days=days)
        except ValueError:
            pass
    
    # Handle absolute dates
    formats = ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    
    raise ValueError(f"Invalid date format: {date_string}. Use YYYY-MM-DD format or relative format like '30d'.")


def main():
    """Main function to run the PR velocity analytics tool"""
    parser = argparse.ArgumentParser(
        description="Generate GitHub PR velocity analytics with comprehensive metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pr_velocity_analytics.py --repo owner/repo --tags "frontend,bug-fix"
  python pr_velocity_analytics.py --repo owner/repo --tags "feature" --group-by author
  python pr_velocity_analytics.py --repo owner/repo --tags "hotfix" --since 30d
  python pr_velocity_analytics.py --repo owner/repo --tags "backend" --since 2024-01-01 --output custom.csv
  
Environment Variables:
  GITHUB_TOKEN - Your GitHub personal access token (required)
        """
    )
    
    parser.add_argument("--repo", "--repository", 
                       required=True,
                       help="GitHub repository in format 'owner/repo'")
    
    parser.add_argument("--tags", "--labels",
                       required=True, 
                       help="Comma-separated list of PR labels/tags to filter by")
    
    parser.add_argument("--since",
                       help="Filter PRs created since date (YYYY-MM-DD format or '30d' for relative)")
    
    parser.add_argument("--group-by",
                       choices=['author'],
                       help="Group results by author")
    
    parser.add_argument("--output", "--output-file",
                       help="Custom output CSV filename (default: auto-generated with date prefix)")
    
    parser.add_argument("--token",
                       help="GitHub personal access token (or set GITHUB_TOKEN env var)")
    
    parser.add_argument("--max-workers", type=int, default=10,
                       help="Maximum number of concurrent API workers (default: 10)")
    
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose output for detailed logging")
    
    args = parser.parse_args()
    
    # Setup logging based on verbose flag
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Get GitHub token
    github_token = args.token or os.getenv("GITHUB_TOKEN")
    if not github_token:
        logger.error("GitHub token is required. Set GITHUB_TOKEN environment variable or use --token")
        logger.info("Get your token at: https://github.com/settings/tokens")
        sys.exit(1)
    
    # Parse configuration
    tags = [tag.strip() for tag in args.tags.split(',')]
    
    since_date = None
    if args.since:
        try:
            since_date = parse_date(args.since)
        except ValueError as e:
            logger.error(f"Error: {e}")
            sys.exit(1)
    
    config = AnalyticsConfig(
        github_token=github_token,
        repo_name=args.repo,
        tags=tags,
        since_date=since_date,
        group_by_author=args.group_by == 'author',
        output_filename=args.output
    )
    
    # Validate max_workers
    max_workers = max(1, min(args.max_workers, 20))  # Limit between 1-20
    if max_workers != args.max_workers:
        logger.info(f"Adjusted max workers from {args.max_workers} to {max_workers}")
    
    try:
        # Initialize components with enhanced error handling
        logger.info(f"Initializing GitHub PR velocity analytics for {config.repo_name}...")
        
        try:
            api_client = EnhancedGitHubAPIClient(config.github_token)
            repo = api_client.get_repository(config.repo_name)
        except GitHubAPIError as e:
            logger.error(f"GitHub API Error: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Initialization Error: {e}")
            sys.exit(1)
        
        logger.info(f"Filtering PRs by tags: {', '.join(config.tags)}")
        if config.since_date:
            logger.info(f"Analyzing PRs since: {config.since_date.strftime('%Y-%m-%d')}")
        
        # Collect PR data with comprehensive error handling using concurrent processing
        all_metrics = []
        latest_pr_date = None
        error_count = 0
        
        try:
            # Get all PRs and filter by tags
            logger.info("🔍 Fetching pull requests from repository...")
            pulls = repo.get_pulls(state='all', sort='created', direction='desc')
            
            # Filter PRs by tags and date in a single pass
            filtered_prs = []
            logger.info("🏷️  Filtering PRs by tags and date criteria...")
            
            for pr in pulls:
                try:
                    # Check if PR matches tag filter
                    pr_labels = [label.name for label in pr.labels] if pr.labels else []
                    if not any(tag in pr_labels for tag in config.tags):
                        continue
                    
                    # Check date filter
                    if config.since_date and pr.created_at and pr.created_at < config.since_date:
                        continue
                    
                    # Track latest PR date for filename
                    if pr.created_at and (latest_pr_date is None or pr.created_at > latest_pr_date):
                        latest_pr_date = pr.created_at
                    
                    filtered_prs.append(pr)
                        
                except AttributeError as e:
                    error_count += 1
                    logger.warning(f"Malformed PR object: {e}")
                    continue
                except Exception as e:
                    error_count += 1
                    logger.warning(f"Error accessing PR: {e}")
                    continue
            
            logger.info(f"✅ Found {len(filtered_prs)} PRs matching criteria")
            
            if not filtered_prs:
                logger.error("No PRs found matching the specified criteria.")
                logger.info("- Check that the repository contains PRs with the specified tags")
                logger.info("- Verify your GitHub token has access to the repository")
                logger.info("- Consider adjusting the --since date filter")
                sys.exit(1)
            
            # Process PRs concurrently
            max_workers = min(10, len(filtered_prs))  # Limit concurrent workers
            results = api_client.process_prs_concurrently(repo, filtered_prs, max_workers)
            
            # Filter out None results (failed processing)
            all_metrics = [result for result in results if result is not None]
            error_count = len([result for result in results if result is None])
            
        except GithubException as e:
            logger.error(f"Error fetching PRs from repository: {e.data.get('message', 'Unknown error')}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error during PR collection: {e}")
            sys.exit(1)
        
        logger.info(f"Collected data for {len(all_metrics)} PRs matching criteria")
        if error_count > 0:
            logger.warning(f"Encountered {error_count} errors during processing")
        
        if not all_metrics:
            logger.error("No PRs found matching the specified criteria.")
            logger.info("- Check that the repository contains PRs with the specified tags")
            logger.info("- Verify your GitHub token has access to the repository")
            logger.info("- Consider adjusting the --since date filter")
            sys.exit(1)
        
        # Generate statistical summaries including communication delay metrics
        metric_names = [
            'cycle_time_hours', 'first_response_time_hours', 'rework_time_hours', 'merge_time_hours',
            'avg_reviewer_to_author_hours', 'avg_author_to_reviewer_hours', 
            'max_communication_gap_hours', 'communication_rounds_count'
        ]
        summaries = {}
        
        for metric_name in metric_names:
            summaries[metric_name] = StatisticalAnalyzer.generate_metric_summary(all_metrics, metric_name)
        
        # Generate output filename
        if config.output_filename:
            output_filename = config.output_filename
        else:
            date_prefix = latest_pr_date.strftime('%Y-%m-%d') if latest_pr_date else datetime.now().strftime('%Y-%m-%d')
            output_filename = f"{date_prefix}_velocity_metrics.csv"
        
        # Export to CSV
        logger.info(f"Exporting results to {output_filename}...")
        
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            # Write individual PR metrics
            fieldnames = [
                'PR Number', 'Title', 'Author', 'Labels', 'Created',
                'Cycle Time (hrs)', 'First Response (hrs)', 'Rework Time (hrs)', 'Merge Time (hrs)',
                'Avg Reviewer to Author Time (hrs)', 'Avg Author to Reviewer Time (hrs)',
                'Max Communication Gap (hrs)', 'Communication Rounds'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Group by author if requested
            if config.group_by_author:
                # Sort by author, then by creation date
                sorted_metrics = sorted(all_metrics, key=lambda x: (x.author, x.created_at))
            else:
                # Sort by creation date (newest first)
                sorted_metrics = sorted(all_metrics, key=lambda x: x.created_at, reverse=True)

            for metrics in sorted_metrics:
                writer.writerow({
                    'PR Number': metrics.pr_number,
                    'Title': metrics.title,
                    'Author': metrics.author,
                    'Labels': ', '.join(metrics.labels),
                    'Created': metrics.created_at.strftime('%Y-%m-%d %H:%M'),
                    'Cycle Time (hrs)': f"{metrics.cycle_time_hours:.2f}" if metrics.cycle_time_hours is not None else "NULL",
                    'First Response (hrs)': f"{metrics.first_response_time_hours:.2f}" if metrics.first_response_time_hours is not None else "NULL",
                    'Rework Time (hrs)': f"{metrics.rework_time_hours:.2f}" if metrics.rework_time_hours is not None else "NULL",
                    'Merge Time (hrs)': f"{metrics.merge_time_hours:.2f}" if metrics.merge_time_hours is not None else "NULL",
                    'Avg Reviewer to Author Time (hrs)': f"{metrics.avg_reviewer_to_author_hours:.2f}" if metrics.avg_reviewer_to_author_hours is not None else "NULL",
                    'Avg Author to Reviewer Time (hrs)': f"{metrics.avg_author_to_reviewer_hours:.2f}" if metrics.avg_author_to_reviewer_hours is not None else "NULL",
                    'Max Communication Gap (hrs)': f"{metrics.max_communication_gap_hours:.2f}" if metrics.max_communication_gap_hours is not None else "NULL",
                    'Communication Rounds': metrics.communication_rounds_count if metrics.communication_rounds_count is not None else "NULL"
                })

            # Write summary statistics
            writer.writerow({})  # Empty row
            writer.writerow({'PR Number': 'SUMMARY STATISTICS'})
            writer.writerow({})

            # Write metric summaries
            summary_fieldnames = ['Metric', 'Count', 'P50 (Median)', 'P75', 'P90', 'P99', 'Mean', 'Std Dev']
            summary_writer = csv.DictWriter(csvfile, fieldnames=summary_fieldnames)
            summary_writer.writeheader()

            for metric_name, summary in summaries.items():
                summary_writer.writerow({
                    'Metric': summary.metric_name,
                    'Count': summary.count,
                    'P50 (Median)': f"{summary.p50_median:.2f}",
                    'P75': f"{summary.p75:.2f}",
                    'P90': f"{summary.p90:.2f}",
                    'P99': f"{summary.p99:.2f}",
                    'Mean': f"{summary.mean:.2f}",
                    'Std Dev': f"{summary.std_dev:.2f}"
                })

            # Write analysis metadata
            writer.writerow({})
            writer.writerow({'PR Number': 'ANALYSIS METADATA'})
            writer.writerow({'PR Number': f"Repository: {config.repo_name}"})
            writer.writerow({'PR Number': f"Tags Filtered: {', '.join(config.tags)}"})
            if config.since_date:
                writer.writerow({'PR Number': f"Since Date: {config.since_date.strftime('%Y-%m-%d')}"})
            writer.writerow({'PR Number': f"Total PRs Analyzed: {len(all_metrics)}"})
            writer.writerow({'PR Number': f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"})

            logger.info(f"Analysis complete! Results exported to {output_filename}")

            # Display console summary with communication delay metrics
            logger.info("\n📊 VELOCITY METRICS SUMMARY")
            logger.info("=" * 50)

            for metric_name, summary in summaries.items():
                if summary.count > 0:
                    logger.info(f"{summary.metric_name}:")
                    logger.info(f"  Count: {summary.count}")
                    logger.info(f"  P50 (Median): {summary.p50_median:.2f} hours")
                    logger.info(f"  P75: {summary.p75:.2f} hours")
                    logger.info(f"  P90: {summary.p90:.2f} hours")
                    logger.info(f"  P99: {summary.p99:.2f} hours")
                    logger.info(f"  Mean: {summary.mean:.2f} hours")
                    logger.info(f"  Std Dev: {summary.std_dev:.2f} hours")
                    logger.info("")

            if config.group_by_author:
                logger.info("📈 Results grouped by author in CSV output")

            logger.info(f"📁 Complete analysis saved to: {output_filename}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()