import logging
from typing import Optional

from azure.core.exceptions import AzureError, ResourceExistsError
from azure.storage.queue.aio import QueueServiceClient
from azure.storage.queue import QueueMessage

from src.core.config import settings

logger = logging.getLogger(__name__)

class StorageQueueService:
    """
    A service class for interacting with Azure Storage Queues.
    Follows a connection and initialization pattern for use in applications like FastAPI.
    """
    def __init__(self, client: QueueServiceClient):
        self.client = client # the service
        self.queue_client = None # the queue

    @classmethod
    def connect(cls):
        """Creates the main QueueServiceClient from the connection string."""
        logger.info("Connecting to Azure Storage Queues...")
        proxies = {}

        if settings.HTTP_PROXY:
            proxies['http_proxy'] = settings.HTTP_PROXY
        if settings.HTTPS_PROXY:
            proxies['https_proxy'] = settings.HTTPS_PROXY

        if proxies:
            logger.info(f'Using proxy configuration: {proxies}')
            client = QueueServiceClient.from_connection_string(settings.AZURE_STORAGE_CONNECTION_STRING, proxies=proxies)
        else:
            client = QueueServiceClient.from_connection_string(settings.AZURE_STORAGE_CONNECTION_STRING)

        logger.info("Storage Queue client created.")
        return cls(client)

    async def initialize(self):
        """
        Idempotently creates the storage queue if it doesn't exist and
        initializes the queue-specific client.
        """
        try:            
            self.queue_client = self.client.get_queue_client(settings.QUEUE_NAME)            
            await self.queue_client.create_queue()
            logger.info(f"Queue '{settings.QUEUE_NAME}' created successfully.")
        except ResourceExistsError:
            logger.info(f"Queue '{settings.QUEUE_NAME}' already exists. Initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Storage Queue: {e}", exc_info=True)
            raise

    async def close(self):
        """Closes the underlying client connection."""
        logger.info("Closing Storage Queue client.")
        await self.client.close()

    async def send_message(self, message_content: str):
        """
        Sends a message to the queue.

        Args:
            message_content: The string content of the message (e.g., a job_id).
        """
        if not self.queue_client:
            raise RuntimeError("Queue client has not been initialized. Call initialize() first.")
        
        try:
            logger.info(f"Sending message to queue '{settings.QUEUE_NAME}'...")
            await self.queue_client.send_message(message_content)
            logger.info(f"Successfully sent message: {message_content}")
        except AzureError as e:
            logger.error(f"Failed to send message to queue '{settings.QUEUE_NAME}': {e}", exc_info=True)
            raise

    async def receive_message(self, visibility_timeout: int = 300) -> Optional[QueueMessage]:
        """
        Receives a single message from the queue, making it invisible for a specified time.

        Args:
            visibility_timeout: The time in seconds that the message should be invisible
                                to other consumers. Defaults to 300 (5 minutes).

        Returns:
            A QueueMessage object if a message is available, otherwise None.
        """
        if not self.queue_client:
            raise RuntimeError("Queue client has not been initialized. Call initialize() first.")

        try:
            logger.debug(f"Attempting to receive message from queue '{settings.QUEUE_NAME}'...")
            message = await self.queue_client.receive_message(visibility_timeout=visibility_timeout)
            if message:
                logger.info(f"Received message with id '{message.id}'.")
            return message
        except AzureError as e:
            logger.error(f"Failed to receive message from queue '{settings.QUEUE_NAME}': {e}", exc_info=True)
            raise

    async def delete_message(self, message: QueueMessage):
        """
        Deletes a message from the queue, typically after it has been successfully processed.

        Args:
            message: The QueueMessage object to delete.
        """
        if not self.queue_client:
            raise RuntimeError("Queue client has not been initialized. Call initialize() first.")
        
        try:
            logger.info(f"Deleting message with id '{message.id}' from queue...")
            await self.queue_client.delete_message(message)
            logger.info(f"Successfully deleted message id '{message.id}'.")
        except AzureError as e:
            # Handle cases where the message might not exist anymore (e.g., visibility timed out
            # and another worker processed and deleted it). This is not necessarily a critical error.
            if e.error_code == "MessageNotFound":
                 logger.warning(f"Could not delete message with id '{message.id}': Message not found. It may have been processed by another worker.")
            else:
                logger.error(f"Failed to delete message with id '{message.id}': {e}", exc_info=True)
                raise