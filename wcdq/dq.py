#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import BaseDQ
from dateutil.parser import parse as dparse
import pytz
from datetime import datetime

"""
This file should implement common DQ requirements.
"""

class DateDQ(BaseDQ):
    """ Data quality for Date types """
    
    def __init__(self, value, doc_data):
        self.value = value
        self.doc_data = doc_data
        super(DateDQ, self).__init__(value, doc_data) 
    
    def get_coherence_score(self):
        """ Date is not coherent if it's in the future or if it's invalid
        """
        if self.get_completeness_score() == 0:
            return 1
        # Invalid date
        try:
            dparsed = dparse(self.value)
            if dparsed.tzinfo is None:
                dparsed = pytz.UTC.localize(dparsed)
        except:
            return 0
        # Date in the future
        if dparsed > datetime.now(pytz.utc):
            return 0
        return 1