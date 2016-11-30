#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from dateutil.parser import parse as dparse

"""
wcdq.py

Data Quality for Web Crawlers

@author: Luiz Fernando de Frias (l.frias@poli.ufrj.br)
"""

class BaseDQ:
    """Base class for DQ score functions.
    
    This class should implement get_<dq_dimension>_score
    for every DQ dimension. Each of this methods should
    return a score between 0 and 1.
    """
    
    def __init__(self, value, doc_data):
        self.value = value
        self.doc_data = doc_data
    
    def get_completeness_score(self):
        """ Default is binary completeness """
        if not self.value:
            return 0
        return 1
    
    def get_coherence_score(self):
        """ Assume value is always consistent """
        return 1


class Document:
    """Abstraction for Document. It'll compute quality
    score for each entry in the attributes mapping.
    """
    
    def __init__(self, did, data, attrs):
        """Constructor method.
        
        Parameters
        ----------

        did:
            document identifier
        data: dictionary
            dictionary with documents attributes
        attrs: dictionary
            Mapping between document attribute, that should be
        in `data` dictionary and its DQ properties: the DQ class
        and its importance to the dimension
            {
                <attribute_name>: {
                    'class': <DQClass (inherits from BaseDQ)>,
                    'weight': <weight (int)>
                },
                ...
            }
        
        => For instance, this is how you'd create a Document instance
            
        >>> d = Document(
                # Document id can be a random string, the database
                # ID (better) or any identifier that you want. 
                did,
                # Document data
                {
                    'title': 'This is a document',
                    'date_published': '2016-11-04T20:00',
                    'body': 'Body for document',
                    ... # 
                },
                # Attributes mapping: should map each attribute in Document
                # data to the class responsible for its data quality
                {
                    'title': {
                        'class': dq.SmallStrDQ,
                        'weight': 2
                    },
                    'date_published': {
                        'class': dq.DateDQ,
                        'weight': 2
                    },
                    'body': {
                        'class': dq.BodyDQ,
                        'weight': 2
                    },
                    ... # for all attributes in Document data
                }
            )
        """

        # Set document ID
        self._id = did 

        # Set attrs for document
        self.attrs_names = list(attrs.keys())
        # Get class to handle DQ quality scores
        # for each attribute
        self.attrs_class = [
            attrs[attr]['class'] for attr in attrs]
        # Get attrs weigh
        self.attrs_weight = [
            attrs[attr]['weight'] for attr in attrs]
        self.attrs = list(zip(self.attrs_names, self.attrs_class))

        # Set data
        self.data = data

        # Set attributes weights
        self._set_weights()

    def __repr__(self):
        """ String representation """
        return 'Document "%s"' % self._id
    
    def _set_weights(self):
        """ Set weights for each attribute. """
        cols = ['attr', 'weight']
        data = []
        for i in range(len(self.attrs_names)):
            data.append(
                [self.attrs_names[i],
                self.attrs_weight[i]]
            )
        self.weights_df = pd.DataFrame(data=data, columns=cols)
    
    def as_df(self):
        """Document as a DataFrame
        
        Returns
        -------
        
        df: DataFrame
            Each column corresponds to an attribute
        """
        data = [self._id]
        data.extend([self.data[attr] for attr in self.attrs_names])
        df = pd.DataFrame(
            data=[data],
            columns=['id'] + self.attrs_names,
        )
        return df.set_index(['id'])
    
    def set_dq(self, dimensions):
        """Compute score for each DQ dimension.
        
        Parameters
        ----------
        
        dimensions: list
            List of DQ dimensions. Each dimension score function
            should be implemented by attribute's DQ class.
        """
        cols = ['attr', 'value']
        cols.extend([dim for dim in dimensions])
        data = []
        for attr, aclass in self.attrs:
            value = self.data[attr]
            aobj = aclass(value, self.data)
            _data = [attr, value,]
            _data.extend(
                [getattr(aobj, 'get_%s_score'% dim)() for dim in dimensions]
            )
            data.append(_data)
        df = pd.DataFrame(data=data, columns=cols)
        self.dq = df

    def describe(self):
        """Describe document as a container of attributes.
        
        Returns
        -------
        
        df: DataFrame
            For each attribute: value, weight, all DQ dimensions scores
        """
        df = self.dq
        weights_df = self.weights_df
        df = pd.merge(df, weights_df, on=['attr'])
        return df
    
    def get_dq(self, as_df=True):
        """Get scores for each DQ dimension.
        
        Parameters
        ----------
        
        as_df: boolean (optional, default is True)
            Return as DataFrame if True
        """
        dq = {}
        df = self.describe()
        df.drop(['attr', 'value'], axis=1, inplace=True)
        for col in df:
            if col == 'weight':
                continue
            score = df[col]*df['weight']/sum(df['weight'])
            dq[col] = score.sum()
        if as_df:
            return pd.DataFrame(dq, index=[0])
        return dq

        
class Spider:
    """Compute DQ dimensions for each Document crawled by
    this Spider.
    """
    
    def __init__(self, spider_id):
        """ Constructor method """
        self._id = spider_id
        self.ndocs = 0
    
    def __repr__(self):
        """ String representation """
        return 'Spider "%s"' % self._id
 
    def as_df(self, docs):
        """ Get Spider as DataFrame.
        
        Parameters
        ----------
        
        docs: generator
            Generator of Documents
        """
        return pd.concat([doc.as_df() for doc in docs])

    def set_dq(self, dimensions, docs):
        """ Compute score for each DQ dimension.
        
        Parameters
        ----------
        
        dimensions: list
            List of DQ dimensions. Each dimension score is an aggregate
            of Document's scores.
        docs: generator
            Generator of Documents
        """
        self.dq_dimensions = dimensions
        cols = ['doc_id']
        cols.extend(dimensions)
        data = []
        ts_data = []
        ts_attrs = None

        dq_by_attrs = None
        
        for idoc, doc in enumerate(docs):
            # Set Document dimensions
            doc.set_dq(dimensions)

            # Create columns for the DQ DataFrame
            # First one is the doc id
            _data = [doc._id]
            # Add a column for every DQ dimension
            dq = doc.get_dq(as_df=False)
            _data.extend([dq[dim] for dim in dimensions])
            data.append(_data)
            
            # Create time series for documents
            _ts_data = {
                'date_retrieved': doc.data.get('date_retrieved'),
            }
            for dim in dimensions:
                _ts_data[dim] = dq[dim]
            ts_data.append(_ts_data)
            
            # Create DQ DataFrame by Document attribute
            # Get docs attributes and scores
            doc_attr_df = doc.describe().set_index(['attr'])
            
            # if it has not been created, fill with 0s
            if dq_by_attrs is None:
                dq_by_attrs = pd.DataFrame(
                    0,
                    index=doc_attr_df.index,
                    columns=dimensions
                )
            dq_by_attrs += doc_attr_df[dimensions]

            # Create time series for DQ dimensions
            _ts_attrs = doc.describe()
            _ts_attrs['date_retrieved'] = doc.data.get('date_retrieved')
            if ts_attrs is None:
                ts_attrs = _ts_attrs
            else:
                ts_attrs = pd.concat([ts_attrs, _ts_attrs])
  
        self.ndocs = idoc + 1

        # DQ DataFrame
        df = pd.DataFrame(data=data, columns=cols)
        self._dq = df
        
        # Take the mean
        dq_by_attrs[dimensions] /= self.ndocs 
        dq_by_attrs.reset_index(inplace=True)
        self._dq_by_attrs = dq_by_attrs
        
        # Time series
        ts_df = pd.DataFrame(ts_data)
        ts_df["date_retrieved"] = ts_df["date_retrieved"].apply(lambda x: dparse(x))
        self.ts_df = ts_df
        # Time series attribute
        ts_attrs.drop(['value', 'weight'], axis=1, inplace=True)
        ts_attrs["date_retrieved"] = ts_attrs["date_retrieved"].apply(lambda x: dparse(x))
        self.ts_attrs_df = ts_attrs
    
    def describe(self, order_by=None, ascending=False):
        """ Return all Documents for Spider.
        
        Parameters
        ----------
        
        order_by: string name or list of names which
        refer to the axis items
        
        ascending: bool or list of bool
        """
        df = self._dq
        if order_by is None:
            return df
        return df.sort_values(order_by, ascending=ascending)
    
    def get_ts(self, freq="20Min"):
        """ Time series for Document retrieved.
        
        Parameters
        ---------
        
        freq: Pandas frequency
        """
        return self.ts_df.groupby(
            pd.Grouper(
                key="date_retrieved", freq=freq
            )
        ).size()
    
    def get_dq_ts(self):
        """ Time series for DQ dimensions""" 
        return self.ts_df
    
    def get_dq_attrs_ts(self):
        """ Time series for DQ dimensions, for attribute""" 
        return self.ts_attrs_df

    def get_dq(self, order_by=None, ascending=True, as_df=True):
        """Get scores for each DQ dimension.
        
        Parameters
        ----------
        
        as_df: boolean (optional, default is True)
            Return as DataFrame if True
        order_by: string name or list of names which
        refer to the axis items
            DataFrame parameter, took the description from
            there
        ascending: bool or list of bool
        """
        dq = {}
        df = self.describe()
        df = df.drop(['doc_id'], axis=1)
        for col in df:
            dq[col] = df[col].mean()
        if not as_df:
            return dq
        # We're suppose to return a DataFrame
        dq_df = pd.DataFrame(dq, index=[0])
        if order_by is None:
            return dq_df
        return dq_df.sort_values(order_by, ascending=ascending)

    def get_dq_by_attrs(self, order_by=None, ascending=True, as_df=True):
        """Get scores for each DQ dimension.
        
        Parameters
        ----------
        
        as_df: boolean (optional, default is True)
            Return as DataFrame if True
        order_by: string name or list of names which
        refer to the axis items
            DataFrame parameter, took the description from
            there
        ascending: bool or list of bool
        """
        df = self._dq_by_attrs
        if not as_df:
            return df.to_dict()
        if order_by is None:
            return df
        return df.sort_values(order_by, ascending=ascending)

    def plot_dq_evolution_for_attr(self, dq_dimension, attr, freq='1H', figsize=None):
        """ Plot score evolution for the given dq dimension and attribute.
        
        Args:
            dq_dimension: str
                DQ dimension
            attr: str
                Which attribute you want.
            freq: str (optiona, default is 1H)
                See Pandas frequencies.
            figsize: tuple (optional, default is None)
                Size of the figure.
        """
        df = self.get_dq_attrs_ts().set_index(['date_retrieved'])
        df = df[df['attr'] == attr]

        ax = df.resample(freq).plot(y=dq_dimension, figsize=figsize)

        ax.set_ylabel(dq_dimension)
        ax.set_xlabel('Date')
        ax.set_ylim([-0.1, 1.1])

        return ax

class DB:
    """ One more degree of granularity. Compute DQ scores
    as an aggregate of Spiders
    """
    
    def __init__(self):
        """ Constructor method """ 
        self.spiders = []
        
    def add_spider(self, spider):
        """ Add Spider.
        
        Parameters
        ----------
        
        spider: Spider
        """
        if spider.ndocs == 0:
            raise Exception(
                'You should call Spider.set_dq before adding it to DB')
        self.spiders.append(spider)
        
    def get_spider(self, spider):
        """ Get Spider.
        
        Parameters
        ----------
        
        spider: string
            Spider ID.
        """
        for s in self.spiders:
            if s._id == spider:
                return s
        raise ValueError("Couldn't find spider %s" % spider)

    def ready(self):
        """ Compute DQ
        """
        n_docs = 0
        data = []
        
        attrs_dq = None

        for s in self.spiders:
            n_docs += s.ndocs
            
            s_scores = s.get_dq(as_df=False)
            spider_data = {
                'spider': s._id,
                'docs': s.ndocs
            }
            for dim in s.dq_dimensions:
                spider_data[dim] = s_scores[dim]
            data.append(spider_data)
            
            s_attrs = s.get_dq_by_attrs().set_index(['attr'])
            s_attrs = s_attrs * s.ndocs
            if attrs_dq is None:
                attrs_dq = s_attrs
            else:
                attrs_dq += s_attrs
            
            s_attrs 
        
        self.ndocs = n_docs 
        self.dq_dimensions = s.dq_dimensions

        cols = ['spider', 'docs'] + self.dq_dimensions
        df = pd.DataFrame(data=data, columns=cols)
        df['weight'] = df['docs']/self.ndocs
        
        attrs_dq /= self.ndocs

        self._df = df
        self._dq_by_attrs = attrs_dq

    def describe(self, order_by='weight', ascending=False):
        """ Return all Spiders with DQ scores.
        
        Parameters
        ----------
        
        order_by: string name or list of names which
        refer to the axis items
        
        ascending: bool or list of bool
        """
        df = self._df.set_index(['spider'])
        if order_by is None:
            return df
        return df.sort_values(order_by, ascending=ascending)

    def get_dq(self, as_df=True):
        """ DQ dimensions.
        
        Parameters
        ----------
        
        as_df: boolean (optional, default is True)
            Return as DataFrame if True
        """
        df = self.describe()
        df.reset_index(inplace=True)
        df = df[self.dq_dimensions].multiply(df['weight'], axis='index')
        self._dq = df[self.dq_dimensions].sum().to_frame().T
        if as_df:
            return self._dq
        return self._dq.to_dict()

    def get_dq_by_attrs(self, order_by=None, ascending=True, as_df=True):
        """Get scores for each DQ dimension.
        
        Parameters
        ----------
        
        as_df: boolean (optional, default is True)
            Return as DataFrame if True
        order_by: string name or list of names which
        refer to the axis items
            DataFrame parameter, took the description from
            there
        ascending: bool or list of bool
        """
        df = self._dq_by_attrs
        if not as_df:
            return df.to_dict()
        if order_by is None:
            return df
        return df.sort_values(order_by, ascending=ascending)

    def plot_dq_by_spider(self, dq_dimension, lower_than=None, greater_than=None, figsize=None):
        """ Plot dq dimension for each spider.
        
        Args:
            dq_dimensions: str
                DQ dimension.
            lower_than: float in [0,1] (optional)
                Wether we should filter.
            greater_than: float in [0,1] (optional)
                Wether we should filter.
            figsize: tuple (default is None)
                Size of the figure.
        """
        df = self.describe(order_by=[dq_dimension], ascending=True)
        if greater_than is not None:
            df = df[df[dq_dimension] > greater_than]
        if lower_than is not None:
            df = df[df[dq_dimension] < lower_than]
        ax = df.reset_index()[['spider', dq_dimension]].plot(kind='bar', x='spider', figsize=(10, 4))
        ax.set_ylabel('%s score' % dq_dimension)
        ax.legend().remove()
        return ax