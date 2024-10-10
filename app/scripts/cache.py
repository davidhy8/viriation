from datetime import datetime
from intervaltree import Interval, IntervalTree
import unittest


class History:
    def __init__(self):
        self.dates = IntervalTree() # Interval search tree with dates that have been screened -> (start date, end date)
        self.papers = {
            'relevant' : set(), # papers that passed screening
            'irrelevant' : set() # papers that were screened out + papers users said were irrelevant
        }


    def checkDateRange(self, date_range):
        """ 
        Determines whether the given date range overlaps with any intervals in the cache of previous scraped dates
        
        Parameters: 
        date_range (tuple): Date range with start date and end date

        Returns:
        bool: Whether the given date range overlaps with any previous date ranges
        """

        start_dt, end_dt = date_range
        start_dt = datetime.strptime(start_dt, '%Y-%m-%d')
        end_dt = datetime.strptime(end_dt, '%Y-%m-%d')
        
        # Convert to timestamp (float) since intervaltree works on numeric values
        start_ts = start_dt.timestamp()
        end_ts = end_dt.timestamp()

        # Query for overlapping intervals in the given range
        overlapping_intervals = self.dates.overlap(start_ts, end_ts)

        return bool(overlapping_intervals)


    def addDateRange(self, date_range):
        """
        Adds new date range into the cache of already scraped dates

        Parameters:
        date_range (tuple): Date range with start date and end date
        """
        start_dt, end_dt = date_range
        start_dt = datetime.strptime(start_dt, '%Y-%m-%d')
        end_dt = end_dt = datetime.strptime(end_dt, '%Y-%m-%d')
        
        start_ts = start_dt.timestamp()
        end_ts = end_dt.timestamp()

        self.dates[start_ts:end_ts] = (start_dt, end_dt) # Add date range
    

    def getNonOverlap(self, date_range):
        """ 
        Returns all dates within the given date range that are not present in the cache of previous scraped dates
        
        Parameters: 
        date_range (tuple): Date range with start date and end date

        Returns:
        list: list of tuples consisting of date ranges that have not been scraped yet
        """
        start_dt, end_dt = date_range
        start_dt = datetime.strptime(start_dt, '%Y-%m-%d')
        end_dt = datetime.strptime(end_dt, '%Y-%m-%d')
        
        # Convert to timestamp (float) since intervaltree works on numeric values
        start_ts = start_dt.timestamp()
        end_ts = end_dt.timestamp()

        # Query for overlapping intervals in the given range
        overlapping_intervals = self.dates.overlap(start_ts, end_ts)

        if not overlapping_intervals:
            return [(start_dt, end_dt)]
        
        overlapping_intervals = sorted(overlapping_intervals)
        non_overlapping_ranges = []
        current_start = start_ts

        # Iterate over each overlapping interval and calculate gaps
        for interval in overlapping_intervals:
            if current_start < interval.begin:
                # There is a gap between the current start and the beginning of this interval
                non_overlapping_ranges.append((current_start, interval.begin))
            # Update current start to the end of the current interval
            current_start = max(current_start, interval.end)
        
        # Check if there's a gap after the last interval
        if current_start < end_ts:
            non_overlapping_ranges.append((current_start, end_ts))
        
        # Convert timestamps back to datetime
        non_overlapping_ranges_dt = [
            (datetime.fromtimestamp(start), datetime.fromtimestamp(end))
            for start, end in non_overlapping_ranges
        ]

        return non_overlapping_ranges_dt


    def updateTree(self):
        """ 
        Merges all overlapping date ranges within the current cache of scraped dates
        """

        self.dates.merge_overlaps() # merge together overlapping intervals
        

    def updatePapers(self, relevant_papers, irrelevant_papers):
        """ 
        Updates history of relevant and irrelevant papers that have been processed through the viriation program thus far
        
        Parameters: 
        relevant_papers (set): Hashset consisting of the DOIs for relevant papers
        irrelevant_papers (set): Hashset consisting of the DOIs for irrelevant papers
        """

        for paper in relevant_papers:
            self.papers['relevant'].add(paper)
        for paper in irrelevant_papers:
            self.papers['irrelevant'].add(paper)
        

    def checkPaper(self, paper):
        """ 
        Checks whether or not a specific paper has been processed by our program before
        
        Parameters: 
        paper (str): DOI of paper

        Returns:
        bool: Whether the paper has been processed by our program before
        """

        return paper in self.papers['relevant'] or paper in self.papers['irrelevant']
    

class TestHistory(unittest.TestCase):

    def setUp(self):
        """Setup a History object and add some initial data."""
        self.history = History()

        # Add initial date ranges
        self.history.addDateRange(('2024-01-01', '2024-01-10'))
        self.history.addDateRange(('2024-02-15', '2024-02-20'))

        # Add some papers
        self.history.updatePapers({'paper_001', 'paper_002'}, {'paper_003', 'paper_004'})

    def test_add_date_range(self):
        """Test adding a date range and checking for overlap."""
        self.history.addDateRange(('2024-03-01', '2024-03-05'))
        # Check if the date range was added properly
        self.assertTrue(self.history.checkDateRange(('2024-03-01', '2024-03-05')))
    
    def test_check_date_range_overlap(self):
        """Test checking if date range overlaps with existing intervals."""
        # This date range overlaps with the first range ('2024-01-01', '2024-01-10')
        self.assertTrue(self.history.checkDateRange(('2024-01-05', '2024-01-12')))

        # This date range does not overlap with any existing ranges
        self.assertFalse(self.history.checkDateRange(('2024-01-11', '2024-01-12')))

    def test_get_non_overlap(self):
        """Test retrieving the non-overlapping part of a date range."""
        # This range overlaps with ('2024-01-01', '2024-01-10'), the non-overlapping part should be '2024-01-10' to '2024-01-12'
        non_overlapping = self.history.getNonOverlap(('2024-01-05', '2024-01-12'))
        expected_non_overlap = [(datetime(2024, 1, 10), datetime(2024, 1, 12))]
        self.assertEqual(non_overlapping, expected_non_overlap)

        # This range doesn't overlap at all, so the entire range should be returned
        non_overlapping = self.history.getNonOverlap(('2024-03-01', '2024-03-10'))
        self.assertEqual(non_overlapping, [(datetime(2024, 3, 1), datetime(2024, 3, 10))])

    # def test_update_tree(self):
        """Test that merging overlapping date ranges works correctly."""
        # Add overlapping date ranges
        self.history.addDateRange(('2024-01-05', '2024-01-15'))

        # Merge overlaps
        self.history.updateTree()

        # Check that the interval has been merged properly
        self.assertTrue(self.history.checkDateRange(('2024-01-01', '2024-01-15')))
        self.assertFalse(self.history.checkDateRange(('2024-01-15', '2024-01-16')))

    def test_update_papers(self):
        """Test updating relevant and irrelevant papers."""
        relevant_papers = {'paper_005', 'paper_006'}
        irrelevant_papers = {'paper_007'}

        # Update paper sets
        self.history.updatePapers(relevant_papers, irrelevant_papers)

        # Check if the papers were added correctly
        self.assertIn('paper_005', self.history.papers['relevant'])
        self.assertIn('paper_007', self.history.papers['irrelevant'])

    def test_check_paper(self):
        """Test checking if a paper has been processed."""
        # Check for a relevant paper
        self.assertTrue(self.history.checkPaper('paper_001'))
        # Check for an irrelevant paper
        self.assertTrue(self.history.checkPaper('paper_003'))
        # Check for a paper not in the set
        self.assertFalse(self.history.checkPaper('paper_008'))

if __name__ == '__main__':
    unittest.main()
