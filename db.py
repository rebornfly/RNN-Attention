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

    #创建订单
    def update_scores(self, skuid, score):
        try:
            #插入base表
            sql = '''
                INSERT INTO t_scores(
                    skuid,
                    score
                )
                values(
                    %s,
                    %s
                )
            '''
            self.execute(sql, skuid, score)

            self.db.commit()
        except  MySQLdb.Error as e:
            logger.info('[ update score ] err:%s', e)
            raise MySQLdb.Error(e)

