import logging
import sys

def setup_logging():
    """全局日志配置，在应用启动时调用一次"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            # 后续可以加 logging.FileHandler("app.log")
        ]
    )