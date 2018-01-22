# Copyright (c) 2017 Lightricks. All rights reserved.
import sys, signal


def create_handle_term():
    def handle_term(signum):
        print("Job Terminated with signal %d" % signum)
        sys.stdout.flush()
        exit(1)

    return handle_term


def set_sigterm_handler():
    signal.signal(signal.SIGTERM, create_handle_term())