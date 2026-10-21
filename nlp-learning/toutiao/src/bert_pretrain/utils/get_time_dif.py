import time
from datetime import timedelta


def get_time_dif(start_time):
    """获取已使用时间，返回格式化的 timedelta 对象 (如 0:01:23)"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=round(time_dif))