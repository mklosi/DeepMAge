import os
import platform
import resource

import psutil


class Memory:
    def __init__(self, noop=False):
        self.process = psutil.Process(os.getpid())
        self.noop = noop
        self.prev = 0

        # Check the platform
        os_name = platform.system()
        # For macOS
        if os_name == "Darwin":
            self.divider = 1024 * 1024
        # For Linux
        elif os_name == "Linux":
            self.divider = 1024
        else:
            raise NotImplementedError(
                f"OS '{os_name}' not supported for this operation."
            )

    def log_memory(self, logger_func, id_string):
        if self.noop:
            return

        psutil_memory_usage = self.process.memory_info().rss  # in bytes
        logger_func(
            f"{id_string} psutil_memory_usage:   {psutil_memory_usage / (1024 * 1024):>10.2f} MB"
        )

        diff = psutil_memory_usage - self.prev
        logger_func(
            f"{id_string} psutil_diff_prev:   {diff / (1024 * 1024):>13.2f} MB"
        )
        self.prev = psutil_memory_usage

        resource_memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On Linux, ru_maxrss is in kilobytes, so you might need to adjust the calculation accordingly
        logger_func(
            f"{id_string} resource_memory_usage: {resource_memory_usage / self.divider:>10.2f} MB"
        )

        avail_memory = psutil.virtual_memory().available
        avail_memory_mb = avail_memory / (1024 * 1024)
        logger_func(f"{id_string} avail_memory_mb:       {avail_memory_mb:>10.2f} MB")
