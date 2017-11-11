#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File   : logger.py
# Author : lightxue
# Date   : 2015-10-20
# Desc   :


import yaml

import logging, logging.config
log_conf = yaml.load(open('logger.yaml'))
logging.config.dictConfig(log_conf)

logger = logging.getLogger('access')
