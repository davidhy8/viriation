import pytest
from datetime import datetime
from scripts.history import History  # Import your class

@pytest.fixture
def history():
    return History()

def test_add_date_range(history):
    # Test adding a date range
    history.addDateRange(('2024-01-01', '2024-01-10'))
    start_ts = datetime.strptime('2024-01-01', '%Y-%m-%d').timestamp()
    end_ts = datetime.strptime('2024-01-10', '%Y-%m-%d').timestamp()
    assert (start_ts, end_ts) in [(i.begin, i.end) for i in history.dates]

def test_check_date_range_overlap(history):
    # Add a date range to the history
    history.addDateRange(('2024-01-01', '2024-01-10'))

    # Test with overlapping range
    assert history.checkDateRange(('2024-01-05', '2024-01-15')) is True

    # Test with non-overlapping range
    assert history.checkDateRange(('2024-02-01', '2024-02-10')) is False

def test_get_non_overlap(history):
    # Add a date range to the history
    history.addDateRange(('2024-01-01', '2024-01-10'))

    # Test for non-overlapping ranges within a given range
    result = history.getNonOverlap(('2024-01-05', '2024-01-20'))
    assert result == [(datetime.strptime('2024-01-10', '%Y-%m-%d'), datetime.strptime('2024-01-20', '%Y-%m-%d'))]

    # Test for a fully non-overlapping range
    result = history.getNonOverlap(('2024-02-01', '2024-02-10'))
    assert result == [(datetime.strptime('2024-02-01', '%Y-%m-%d'), datetime.strptime('2024-02-10', '%Y-%m-%d'))]

def test_update_tree(history):
    # Add overlapping date ranges to the history
    history.addDateRange(('2024-01-01', '2024-01-10'))
    history.addDateRange(('2024-01-05', '2024-01-15'))

    # Merge the overlapping ranges
    history.updateTree()

    # Check that the merge worked
    intervals = list(history.dates)
    assert len(intervals) == 1  # Should merge into one interval
    assert intervals[0].begin == datetime.strptime('2024-01-01', '%Y-%m-%d').timestamp()
    assert intervals[0].end == datetime.strptime('2024-01-15', '%Y-%m-%d').timestamp()

def test_add_paper(history):
    # Add relevant and irrelevant papers
    history.addPaper('10.1001/jama.2024.12345', True)
    history.addPaper('10.1001/jama.2024.54321', False)

    # Check that papers were added correctly
    assert '10.1001/jama.2024.12345' in history.papers['relevant']
    assert '10.1001/jama.2024.54321' in history.papers['irrelevant']

def test_check_paper(history):
    # Add papers
    history.addPaper('10.1001/jama.2024.12345', True)
    history.addPaper('10.1001/jama.2024.54321', False)

    # Test that the papers are recognized
    assert history.checkPaper('10.1001/jama.2024.12345') is True
    assert history.checkPaper('10.1001/jama.2024.54321') is True

    # Test a paper that has not been added
    assert history.checkPaper('10.1001/jama.2024.99999') is False
