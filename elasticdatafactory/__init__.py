# -*- coding: utf-8 -*-
# importing necessary libraries for logging
import logging
# import handlers
import logging.handlers
# import functions for interacting with the operating system
from os.path import dirname, abspath, join, exists

# this code is for logging, accessing the version, git commit id, and git remote path of the package
# this information stored in text files by the CICD pipeline when the package is built
package_path = dirname(abspath(__file__))

# collect version information
version_path = join(package_path, "version.txt")
if exists(version_path):
    with open(version_path) as f:
        __version__ = f.read().strip()
else:
    __version__ = "N/A"
    print("No version file found!")

# collect git commit information
git_id_path = join(package_path, "git_commit_id.txt")
if exists(git_id_path):
    with open(git_id_path) as f:
        __git_commit_id__ = f.read()
else:
    __git_commit_id__ = "N/A"

# collect git remote information
git_remote_path = join(package_path, "git_remote.txt")
if exists(git_remote_path):
    with open(git_remote_path) as f:
        __git_remote__ = f.read()
else:
    __git_remote__ = "N/A"

# define logger for the module and submodules
def setup_custom_logger(name):
    # logger settings
    log_file = './logger.log'
    log_file_max_size = 1024 * 1024 * 20 # megabytes
    log_num_backups = 3
    log_format = '%(asctime)s [%(levelname)s] %(filename)s/%(funcName)s:%(lineno)s >> %(message)s'
    log_filemode = 'a' # w: overwrite; a: append
    
    # setup logger
    logging.basicConfig(filename=log_file, format=log_format,
                        filemode=log_filemode, level=logging.INFO)
    rotate_file = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=log_file_max_size, backupCount=log_num_backups
    )
    logger = logging.getLogger(name)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.addHandler(rotate_file)

    # print log messages to console
    consoleHandler = logging.StreamHandler()
    logFormatter = logging.Formatter(log_format)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    
    return logger
