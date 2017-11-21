#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : store.py
# Date   : 2015-09-09
# Desc   :

#import MySQLdb
import pymysql as MySQLdb
import config as config
from logger import logger

class DB(object):
    def __init__(self):
        self.connect()
        self.db.ping(True)
        self.cursor = self.db.cursor(MySQLdb.cursors.DictCursor)

    def connect(self):
        self.db = MySQLdb.connect(**config.options['db'])

    def execute(self, sql, *args):
        try:
            self.cursor.execute(sql, args)
        except (AttributeError, MySQLdb.OperationalError) as e:
            logger.info('Connection is gone away, reconnecting: %s', e)
            self.connect()
            self.cursor = self.db.cursor(MySQLdb.cursors.DictCursor)
            self.cursor.execute(sql, args)

