#!/usr/bin/env python3
"""
Unit tests for GitHub PR Velocity Analytics Tool

Tests for velocity metrics calculations, statistical analysis, and data processing.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone, timedelta
import json
import tempfile
import os
import sqlite3

# Import the modules to test
from pr_velocity_analytics import (
    PRVelocityMetrics,
    MetricSummary,
    AnalyticsConfig,
    PRVelocityCache,
    VelocityMetricsCalculator,
    StatisticalAnalyzer,
    parse_date,
    retry_on_github_error,
    GitHubAPIError,
    DataProcessingError
)


class TestPRVelocityMetrics(unittest.TestCase):
    """Test PRVelocityMetrics data class"""
    
    def test_pr_velocity_metrics_initialization(self):
        """Test PRVelocityMetrics initialization"""
        created_at = datetime.now(timezone.utc)
        metrics = PRVelocityMetrics(
            pr_number=123,
            title="Test PR",
            author="testuser",
            labels=["bug", "frontend"],
            created_at=created_at
        )
        
        self.assertEqual(metrics.pr_number, 123)
        self.assertEqual(metrics.title, "Test PR")
        self.assertEqual(metrics.author, "testuser")
        self.assertEqual(metrics.labels, ["bug", "frontend"])
        self.assertEqual(metrics.created_at, created_at)
        self.assertIsNone(metrics.cycle_time_hours)
        self.assertIsNone(metrics.first_response_time_hours)
        self.assertIsNone(metrics.rework_time_hours)
        self.assertIsNone(metrics.merge_time_hours)


class TestPRVelocityCache(unittest.TestCase):
    """Test PRVelocityCache SQLite caching functionality"""
    
    def setUp(self):
        """Set up test cache with temporary database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.cache = PRVelocityCache(self.temp_db.name)
    
    def tearDown(self):
        """Clean up temporary database"""
        os.unlink(self.temp_db.name)
    
    def test_cache_initialization(self):
        """Test cache database initialization"""
        # Check that tables were created
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        self.assertIn('pr_data', tables)
        self.assertIn('pr_reviews', tables)
        self.assertIn('pr_comments', tables)
        
        conn.close()
    
    def test_cache_pr_data_operations(self):
        """Test caching and retrieving PR data"""
        test_data = {
            'pr_number': 123,
            'title': 'Test PR',
            'author': 'testuser',
            'labels': ['bug']
        }
        
        # Test caching
        self.cache.cache_pr_data('owner/repo', 123, test_data)
        
        # Test retrieval
        cached_data = self.cache.get_cached_pr_data('owner/repo', 123)
        self.assertEqual(cached_data, test_data)
        
        # Test non-existent data
        missing_data = self.cache.get_cached_pr_data('owner/repo', 999)
        self.assertIsNone(missing_data)
    
    def test_cache_reviews_operations(self):
        """Test caching and retrieving PR reviews"""
        test_reviews = [
            {'submitted_at': '2024-01-15T10:00:00Z', 'state': 'APPROVED', 'user': 'reviewer1'},
            {'submitted_at': '2024-01-14T15:30:00Z', 'state': 'CHANGES_REQUESTED', 'user': 'reviewer2'}
        ]
        
        # Test caching
        self.cache.cache_reviews('owner/repo', 123, test_reviews)
        
        # Test retrieval
        cached_reviews = self.cache.get_cached_reviews('owner/repo', 123)
        self.assertEqual(cached_reviews, test_reviews)
    
    def test_cache_comments_operations(self):
        """Test caching and retrieving PR comments"""
        test_comments = [
            {'created_at': '2024-01-15T12:00:00Z', 'user': 'commenter1'},
            {'created_at': '2024-01-15T14:00:00Z', 'user': 'commenter2'}
        ]
        
        # Test caching
        self.cache.cache_comments('owner/repo', 123, test_comments)
        
        # Test retrieval
        cached_comments = self.cache.get_cached_comments('owner/repo', 123)
        self.assertEqual(cached_comments, test_comments)


class TestVelocityMetricsCalculator(unittest.TestCase):
    """Test VelocityMetricsCalculator calculations"""
    
    def test_cycle_time_calculation(self):
        """Test cycle time calculation"""
        timeline_data = {
            'pr_number': 123,
            'title': 'Test PR',
            'author': 'testuser',
            'labels': ['feature'],
            'created_at': '2024-01-15T10:00:00Z',
            'merged_at': '2024-01-17T14:00:00Z',
            'reviews': [],
            'comments': []
        }
        
        metrics = VelocityMetricsCalculator.calculate_metrics(timeline_data)
        
        # Cycle time should be 52 hours (2 days + 4 hours)
        self.assertAlmostEqual(metrics.cycle_time_hours, 52.0, places=1)
        self.assertEqual(metrics.pr_number, 123)
        self.assertEqual(metrics.title, 'Test PR')
        self.assertEqual(metrics.author, 'testuser')
    
    def test_first_response_time_calculation(self):
        """Test first response time calculation"""
        timeline_data = {
            'pr_number': 123,
            'title': 'Test PR',
            'author': 'testuser',
            'labels': ['feature'],
            'created_at': '2024-01-15T10:00:00Z',
            'merged_at': '2024-01-17T14:00:00Z',
            'reviews': [
                {'submitted_at': '2024-01-15T14:00:00Z', 'state': 'APPROVED', 'user': 'reviewer1'}
            ],
            'comments': [
                {'created_at': '2024-01-15T12:00:00Z', 'user': 'commenter1'}
            ]
        }
        
        metrics = VelocityMetricsCalculator.calculate_metrics(timeline_data)
        
        # First response should be comment at 12:00 (2 hours after creation)
        self.assertAlmostEqual(metrics.first_response_time_hours, 2.0, places=1)
    
    def test_rework_time_calculation(self):
        """Test rework time calculation"""
        timeline_data = {
            'pr_number': 123,
            'title': 'Test PR',
            'author': 'testuser',
            'labels': ['feature'],
            'created_at': '2024-01-15T10:00:00Z',
            'merged_at': '2024-01-17T14:00:00Z',
            'reviews': [
                {'submitted_at': '2024-01-15T14:00:00Z', 'state': 'CHANGES_REQUESTED', 'user': 'reviewer1'},
                {'submitted_at': '2024-01-16T10:00:00Z', 'state': 'APPROVED', 'user': 'reviewer1'}
            ],
            'comments': []
        }
        
        metrics = VelocityMetricsCalculator.calculate_metrics(timeline_data)
        
        # Rework time should be 20 hours (changes requested at 14:00, approved at 10:00 next day)
        self.assertAlmostEqual(metrics.rework_time_hours, 20.0, places=1)
    
    def test_merge_time_calculation(self):
        """Test merge time calculation"""
        timeline_data = {
            'pr_number': 123,
            'title': 'Test PR',
            'author': 'testuser',
            'labels': ['feature'],
            'created_at': '2024-01-15T10:00:00Z',
            'merged_at': '2024-01-16T12:00:00Z',
            'reviews': [
                {'submitted_at': '2024-01-16T10:00:00Z', 'state': 'APPROVED', 'user': 'reviewer1'}
            ],
            'comments': []
        }
        
        metrics = VelocityMetricsCalculator.calculate_metrics(timeline_data)
        
        # Merge time should be 2 hours (approved at 10:00, merged at 12:00)
        self.assertAlmostEqual(metrics.merge_time_hours, 2.0, places=1)
    
    def test_unmerged_pr_handling(self):
        """Test handling of unmerged PRs"""
        timeline_data = {
            'pr_number': 123,
            'title': 'Test PR',
            'author': 'testuser',
            'labels': ['feature'],
            'created_at': '2024-01-15T10:00:00Z',
            'merged_at': None,
            'reviews': [],
            'comments': []
        }
        
        metrics = VelocityMetricsCalculator.calculate_metrics(timeline_data)
        
        # Cycle time should be None for unmerged PRs
        self.assertIsNone(metrics.cycle_time_hours)
    
    def test_malformed_data_handling(self):
        """Test handling of malformed timeline data"""
        timeline_data = {
            'pr_number': 123,
            'title': 'Test PR',
            'author': 'testuser',
            'labels': ['feature'],
            'created_at': 'invalid-date',
            'merged_at': None,
            'reviews': [],
            'comments': []
        }
        
        # Should not raise exception, but handle gracefully
        metrics = VelocityMetricsCalculator.calculate_metrics(timeline_data)
        self.assertEqual(metrics.pr_number, 123)
        self.assertEqual(metrics.title, 'Test PR')


class TestStatisticalAnalyzer(unittest.TestCase):
    """Test StatisticalAnalyzer percentile calculations"""
    
    def setUp(self):
        """Set up test data with various metrics"""
        self.test_metrics = []
        
        # Create test metrics with known values
        for i, (cycle, response, rework, merge) in enumerate([
            (24.0, 2.0, 8.0, 1.0),    # Fast turnaround
            (48.0, 4.0, 12.0, 2.0),   # Medium turnaround  
            (72.0, 8.0, 16.0, 4.0),   # Slow turnaround
            (96.0, 12.0, 24.0, 6.0),  # Very slow turnaround
            (120.0, 16.0, None, 8.0)  # Slowest, no rework
        ]):
            metrics = PRVelocityMetrics(
                pr_number=i + 1,
                title=f"PR {i + 1}",
                author="testuser",
                labels=["test"],
                created_at=datetime.now(timezone.utc)
            )
            metrics.cycle_time_hours = cycle
            metrics.first_response_time_hours = response
            metrics.rework_time_hours = rework
            metrics.merge_time_hours = merge
            
            self.test_metrics.append(metrics)
    
    def test_percentile_calculation(self):
        """Test percentile calculation"""
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        percentiles = StatisticalAnalyzer.calculate_percentiles(values)
        
        self.assertEqual(percentiles['p50'], 55.0)  # Median
        self.assertEqual(percentiles['p75'], 77.5)
        self.assertEqual(percentiles['p90'], 91.0)
        self.assertEqual(percentiles['p99'], 99.1)
    
    def test_empty_values_handling(self):
        """Test handling of empty values list"""
        percentiles = StatisticalAnalyzer.calculate_percentiles([])
        
        self.assertEqual(percentiles['p50'], 0)
        self.assertEqual(percentiles['p75'], 0)
        self.assertEqual(percentiles['p90'], 0)
        self.assertEqual(percentiles['p99'], 0)
    
    def test_cycle_time_summary_generation(self):
        """Test cycle time metric summary generation"""
        summary = StatisticalAnalyzer.generate_metric_summary(self.test_metrics, 'cycle_time')
        
        self.assertEqual(summary.metric_name, 'Cycle Time')
        self.assertEqual(summary.count, 5)
        self.assertEqual(summary.p50_median, 72.0)  # Middle value
        self.assertGreater(summary.mean, 0)
        self.assertGreater(summary.std_dev, 0)
    
    def test_rework_time_summary_with_nulls(self):
        """Test rework time summary handling None values"""
        summary = StatisticalAnalyzer.generate_metric_summary(self.test_metrics, 'rework_time')
        
        self.assertEqual(summary.metric_name, 'Rework Time')
        self.assertEqual(summary.count, 4)  # One None value should be excluded
        self.assertGreater(summary.mean, 0)
    
    def test_empty_metrics_summary(self):
        """Test summary generation with no valid metrics"""
        empty_metrics = []
        summary = StatisticalAnalyzer.generate_metric_summary(empty_metrics, 'cycle_time')
        
        self.assertEqual(summary.metric_name, 'Cycle Time')
        self.assertEqual(summary.count, 0)
        self.assertEqual(summary.p50_median, 0)
        self.assertEqual(summary.mean, 0)
        self.assertEqual(summary.std_dev, 0)


class TestDateParsing(unittest.TestCase):
    """Test date parsing functionality"""
    
    def test_relative_date_parsing(self):
        """Test relative date parsing (30d, 7d, etc.)"""
        result = parse_date('30d')
        expected = datetime.now(timezone.utc) - timedelta(days=30)
        
        # Should be within 1 second of expected
        self.assertLess(abs((result - expected).total_seconds()), 1.0)
    
    def test_absolute_date_parsing(self):
        """Test absolute date parsing"""
        result = parse_date('2024-01-15')
        expected = datetime(2024, 1, 15, tzinfo=timezone.utc)
        
        self.assertEqual(result, expected)
    
    def test_alternative_date_formats(self):
        """Test alternative date formats"""
        formats_and_dates = [
            ('2024/01/15', datetime(2024, 1, 15, tzinfo=timezone.utc)),
            ('01/15/2024', datetime(2024, 1, 15, tzinfo=timezone.utc)),
            ('15/01/2024', datetime(2024, 1, 15, tzinfo=timezone.utc))
        ]
        
        for date_str, expected in formats_and_dates:
            with self.subTest(date_str=date_str):
                result = parse_date(date_str)
                self.assertEqual(result, expected)
    
    def test_invalid_date_parsing(self):
        """Test invalid date format handling"""
        with self.assertRaises(ValueError):
            parse_date('invalid-date')
        
        with self.assertRaises(ValueError):
            parse_date('2024-13-45')  # Invalid month/day


class TestRetryDecorator(unittest.TestCase):
    """Test retry decorator functionality"""
    
    def test_successful_retry_after_failure(self):
        """Test successful execution after initial failures"""
        call_count = 0
        
        @retry_on_github_error(max_retries=3, backoff_factor=0.1)
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = test_function()
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)
    
    def test_max_retries_exceeded(self):
        """Test behavior when max retries are exceeded"""
        @retry_on_github_error(max_retries=2, backoff_factor=0.1)
        def test_function():
            raise Exception("Persistent failure")
        
        with self.assertRaises(Exception):
            test_function()
    
    def test_no_retry_on_auth_error(self):
        """Test that authentication errors are not retried"""
        from github import GithubException
        
        call_count = 0
        
        @retry_on_github_error(max_retries=3, backoff_factor=0.1)
        def test_function():
            nonlocal call_count
            call_count += 1
            raise GithubException(401, {"message": "Unauthorized"})
        
        with self.assertRaises(GithubException):
            test_function()
        
        self.assertEqual(call_count, 1)  # Should not retry auth errors


class TestAnalyticsConfig(unittest.TestCase):
    """Test AnalyticsConfig data class"""
    
    def test_analytics_config_creation(self):
        """Test AnalyticsConfig initialization"""
        config = AnalyticsConfig(
            github_token="test_token",
            repo_name="owner/repo",
            tags=["bug", "feature"],
            since_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            group_by_author=True,
            output_filename="test.csv"
        )
        
        self.assertEqual(config.github_token, "test_token")
        self.assertEqual(config.repo_name, "owner/repo")
        self.assertEqual(config.tags, ["bug", "feature"])
        self.assertEqual(config.since_date, datetime(2024, 1, 1, tzinfo=timezone.utc))
        self.assertTrue(config.group_by_author)
        self.assertEqual(config.output_filename, "test.csv")


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete workflows"""
    
    def test_complete_metrics_calculation_workflow(self):
        """Test complete workflow from timeline data to statistical summary"""
        # Create realistic timeline data
        timeline_data = {
            'pr_number': 123,
            'title': 'Add user authentication',
            'author': 'developer1',
            'labels': ['feature', 'security'],
            'created_at': '2024-01-15T09:00:00Z',
            'merged_at': '2024-01-18T15:30:00Z',
            'reviews': [
                {'submitted_at': '2024-01-15T11:30:00Z', 'state': 'CHANGES_REQUESTED', 'user': 'reviewer1'},
                {'submitted_at': '2024-01-16T14:00:00Z', 'state': 'APPROVED', 'user': 'reviewer1'},
                {'submitted_at': '2024-01-17T10:00:00Z', 'state': 'APPROVED', 'user': 'reviewer2'}
            ],
            'comments': [
                {'created_at': '2024-01-15T10:15:00Z', 'user': 'commenter1'},
                {'created_at': '2024-01-16T16:00:00Z', 'user': 'developer1'}
            ]
        }
        
        # Calculate metrics
        metrics = VelocityMetricsCalculator.calculate_metrics(timeline_data)
        
        # Verify calculations
        self.assertEqual(metrics.pr_number, 123)
        self.assertEqual(metrics.title, 'Add user authentication')
        self.assertEqual(metrics.author, 'developer1')
        self.assertEqual(metrics.labels, ['feature', 'security'])
        
        # Verify timing calculations
        self.assertIsNotNone(metrics.cycle_time_hours)
        self.assertIsNotNone(metrics.first_response_time_hours)
        self.assertIsNotNone(metrics.rework_time_hours)
        
        # First response should be comment at 10:15 (1.25 hours after creation)
        self.assertAlmostEqual(metrics.first_response_time_hours, 1.25, places=1)
        
        # Generate statistical summary
        metrics_list = [metrics]
        summary = StatisticalAnalyzer.generate_metric_summary(metrics_list, 'cycle_time')
        
        self.assertEqual(summary.count, 1)
        self.assertGreater(summary.p50_median, 0)


if __name__ == '__main__':
    # Set up test environment
    import sys
    sys.path.insert(0, '.')
    
    # Run tests
    unittest.main(verbosity=2)