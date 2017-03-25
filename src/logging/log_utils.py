import logging
import time

def logger(name=None):
    log = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    if not len(log.handlers):
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)
    return log

def method_header(method_name=None, *args):
    """ 
    Get the formatted method header with arguments included
    """
    hdr = "%s(" % method_name
    if (len(args) > 0):
        hdr += args[0]
    for arg in args[1:]:
        hdr += ", %s" % arg
    hdr += ')'
    return hdr

def replace_at_ind(tup=None, ind=None, val=None):
    return tup[:ind] + (val,) + tup[ind+1:]

def print_date_time(log):
    log.info('The date is ' + time.strftime('%m/%d/%Y'))
    log.info('The time is ' + time.strftime('%I:%M:%S %p'))

def _hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def print_elapsed_time(log, start=None, end=None):
    time_elapsed = (end - start)
    log.info("It took {} to train the model".format(_hms_string(time_elapsed)))
