"""
日志管理模块

提供统一的日志管理功能，支持控制台和文件日志输出。

作者：Augment Agent
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from .base import Singleton


class LoggerManager(metaclass=Singleton):
    """
    日志管理器
    
    使用单例模式确保全局日志的一致性
    """
    
    def __init__(self):
        """初始化日志管理器"""
        self._loggers = {}
        self._log_dir = Path(__file__).parent.parent / "logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._default_level = logging.INFO
        self._default_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
    def get_logger(self, name: str, level: Optional[int] = None, 
                  log_file: Optional[str] = None, console: bool = True) -> logging.Logger:
        """
        获取或创建日志记录器
        
        Args:
            name: 日志记录器名称
            level: 日志级别，默认使用INFO
            log_file: 日志文件路径，默认使用名称和时间戳
            console: 是否输出到控制台
            
        Returns:
            日志记录器
        """
        # 如果已存在则返回
        if name in self._loggers:
            return self._loggers[name]
        
        # 创建新的日志记录器
        logger = logging.getLogger(name)
        
        # 设置日志级别
        logger.setLevel(level or self._default_level)
        
        # 避免日志重复
        logger.propagate = False
        
        # 清除现有处理器
        if logger.handlers:
            logger.handlers.clear()
        
        # 创建格式化器
        formatter = logging.Formatter(self._default_format)
        
        # 添加控制台处理器
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 添加文件处理器
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self._log_dir / f"{name}_{timestamp}.log"
        else:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 保存日志记录器
        self._loggers[name] = logger
        
        return logger
    
    def setup_root_logger(self, level: Optional[int] = None, 
                         log_file: Optional[str] = None, console: bool = True):
        """
        设置根日志记录器
        
        Args:
            level: 日志级别
            log_file: 日志文件路径
            console: 是否输出到控制台
        """
        return self.get_logger("root", level, log_file, console)
    
    def setup_pipeline_logger(self, pipeline_name: str, level: Optional[int] = None):
        """
        设置流水线日志记录器
        
        Args:
            pipeline_name: 流水线名称
            level: 日志级别
            
        Returns:
            日志记录器
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self._log_dir / f"{pipeline_name}_{timestamp}.log"
        return self.get_logger(pipeline_name, level, log_file)
    
    def get_all_loggers(self) -> Dict[str, logging.Logger]:
        """获取所有日志记录器"""
        return self._loggers.copy()
    
    def set_default_level(self, level: int):
        """
        设置默认日志级别
        
        Args:
            level: 日志级别
        """
        self._default_level = level
    
    def set_default_format(self, format_str: str):
        """
        设置默认日志格式
        
        Args:
            format_str: 格式字符串
        """
        self._default_format = format_str
    
    def set_log_dir(self, log_dir: str):
        """
        设置日志目录
        
        Args:
            log_dir: 日志目录路径
        """
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
    
    def disable_logger(self, name: str):
        """
        禁用日志记录器
        
        Args:
            name: 日志记录器名称
        """
        if name in self._loggers:
            logger = self._loggers[name]
            logger.handlers.clear()
            logger.addHandler(logging.NullHandler())
            logger.propagate = False
    
    def enable_logger(self, name: str):
        """
        启用日志记录器
        
        Args:
            name: 日志记录器名称
        """
        if name in self._loggers:
            # 重新创建日志记录器
            del self._loggers[name]
            self.get_logger(name)
    
    def get_log_files(self) -> List[str]:
        """获取所有日志文件路径"""
        return [str(f) for f in self._log_dir.glob("*.log")]
    
    def clear_loggers(self):
        """清空所有日志记录器"""
        for logger in self._loggers.values():
            logger.handlers.clear()
        self._loggers.clear()
    
    def get_logger_info(self) -> Dict[str, Dict[str, Any]]:
        """获取所有日志记录器信息"""
        info = {}
        for name, logger in self._loggers.items():
            handlers_info = []
            for handler in logger.handlers:
                handler_type = type(handler).__name__
                if isinstance(handler, logging.FileHandler):
                    handlers_info.append({
                        "type": handler_type,
                        "file": handler.baseFilename
                    })
                else:
                    handlers_info.append({"type": handler_type})
            
            info[name] = {
                "level": logging.getLevelName(logger.level),
                "handlers": handlers_info,
                "propagate": logger.propagate
            }
        
        return info
