"""
Celery application factory for ChatREL v4.
Configures Celery with robust settings for production use.
"""

import logging
from celery import Celery, signals
from chatrel import config
from chatrel.utils.redis_client import get_redis_client

logger = logging.getLogger(__name__)

def make_celery(app_name=__name__):
    """
    Create and configure a Celery application.
    """
    celery_app = Celery(
        app_name,
        broker=config.CELERY_BROKER_URL,
        backend=config.CELERY_RESULT_BACKEND
    )

    # Robust configuration
    celery_app.conf.update(
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        broker_transport_options={
            'visibility_timeout': 3600,
            'max_retries': 3,
            'interval_start': 0,
            'interval_step': 0.2,
            'interval_max': 0.5,
        },
        broker_pool_limit=config.BROKER_POOL_LIMIT,
        broker_connection_retry=config.BROKER_CONNECTION_RETRY,
        broker_connection_max_retries=5,
        broker_connection_retry_on_startup=True,
        task_default_retry_delay=5,
        task_annotations={
            'chatrel.tasks.update_word_stats.update_token_stats_async': {
                'rate_limit': '100/m'
            }
        },
        # Reduce noise
        worker_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
        worker_task_log_format="[%(asctime)s: %(levelname)s/%(processName)s] [%(task_name)s(%(task_id)s)] %(message)s",
    )

    return celery_app

celery = make_celery()

@signals.worker_ready.connect
def log_worker_ready(sender=None, **kwargs):
    """Log when worker is ready."""
    logger.info(f"Celery worker ready: {sender}")
    # Verify Redis connection
    client = get_redis_client()
    if client:
        logger.info("Redis connection verified on worker startup")
    else:
        logger.warning("Redis connection failed on worker startup")

@signals.beat_init.connect
def log_beat_init(sender=None, **kwargs):
    """Log when beat is initialized."""
    logger.info("Celery beat initialized")
