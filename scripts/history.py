from datetime import datetime
from intervaltree import Interval, IntervalTree


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
        

    def addPaper(self, paper, relevance):
        """ 
        Updates history of relevant and irrelevant papers that have been processed through the viriation program thus far
        
        Parameters: 
            relevant_papers (str): DOI of paper
            relevance (bool): Whether or not the paper is relevant
        """

        if relevance:
            self.papers['relevant'].add(paper)
        
        else:
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