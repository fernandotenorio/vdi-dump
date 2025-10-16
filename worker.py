import asyncio
import io
import logging
from datetime import datetime, timezone

from src.core.config import settings
from src.models.job import JobStatus
from src.services.blob_service import BlobStorageService
from src.services.cosmos_service import CosmosDBService
from src.services.queue_service import StorageQueueService
from src.services.gemini_service import GeminiAsyncOCR # Assuming you saved it here

# Configure logging for the worker
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ocr_worker")

# --- Define constants from settings for clarity ---
MAX_RETRIES = 3
VISIBILITY_TIMEOUT = 600  # 10 minutes, should be longer than a single OCR task

async def process_job(
    job_id: str,
    cosmos: CosmosDBService,
    blob: BlobStorageService,
    gemini: GeminiAsyncOCR,
):
    """The core logic for processing a single OCR job."""
    logger.info(f"Processing job_id: {job_id}")
    job_doc = await cosmos.get_job_by_id(job_id)

    if not job_doc:
        logger.error(f"Job {job_id} not found in database. Message will be left on queue and will time out.")
        return False # Indicates failure to process, but not a permanent one

    # 1. Check for Terminal State or Max Retries
    if job_doc.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
        logger.warning(f"Job {job_id} is already in a terminal state '{job_doc.status}'. Acknowledging message.")
        return True # Acknowledge and delete message

    if job_doc.retry_count >= MAX_RETRIES:
        logger.error(f"Job {job_id} has exceeded max retries. Marking as FAILED.")
        job_doc.status = JobStatus.FAILED
        job_doc.error_log.append(f"Processing failed after {MAX_RETRIES} retries.")
        job_doc.updated_at = datetime.now(timezone.utc)
        await cosmos.collection.replace_one({"_id": job_doc.id}, job_doc.dict(by_alias=True))
        return True # Acknowledge and delete message

    # 2. Set status to Processing and Increment Retry Count
    try:
        job_doc.status = JobStatus.PROCESSING
        job_doc.retry_count += 1
        job_doc.updated_at = datetime.now(timezone.utc)
        await cosmos.collection.replace_one({"_id": job_doc.id}, job_doc.dict(by_alias=True))

        # 3. Perform OCR
        pdf_bytes = await blob.download_file(f"{job_id}.pdf")
        if not pdf_bytes:
            raise FileNotFoundError(f"PDF for job {job_id} not found in blob storage.")
        
        pdf_stream = io.BytesIO(pdf_bytes)
        ocr_result = await gemini.run(file=pdf_stream, pages_per_part=job_doc.pages_per_part)

        if ocr_result is None:
            raise RuntimeError("Gemini OCR process returned None, indicating a failure.")

        # 4. Handle Success
        logger.info(f"Successfully performed OCR for job {job_id}.")
        job_doc.status = JobStatus.COMPLETED
        job_doc.final_text = ocr_result
        job_doc.updated_at = datetime.now(timezone.utc)
        await cosmos.collection.replace_one({"_id": job_doc.id}, job_doc.dict(by_alias=True))
        return True # Success, message will be deleted
    
    except Exception as e:
        # 5. Handle Failure
        logger.error(f"An error occurred while processing job {job_id}: {e}", exc_info=True)
        job_doc.status = JobStatus.QUEUED # Reset status for next attempt
        job_doc.error_log.append(f"Attempt {job_doc.retry_count} failed: {str(e)}")
        job_doc.updated_at = datetime.now(timezone.utc)
        await cosmos.collection.replace_one({"_id": job_doc.id}, job_doc.dict(by_alias=True))
        return False # Failure, message will not be deleted and will re-appear

async def main():
    """Main worker loop."""
    logger.info("Starting OCR Worker...")
    
    # Initialize services
    cosmos_service = CosmosDBService.connect()
    blob_service = BlobStorageService.connect()
    await blob_service.initialize()
    queue_service = StorageQueueService.connect()
    await queue_service.initialize()
    gemini_client = GeminiAsyncOCR()
    
    try:
        while True:
            message = await queue_service.receive_message(visibility_timeout=VISIBILITY_TIMEOUT)
            if message:
                job_id = message.content
                is_processed_successfully = await process_job(
                    job_id, cosmos_service, blob_service, gemini_client
                )
                if is_processed_successfully:
                    await queue_service.delete_message(message)
            else:
                logger.info("No messages in queue. Waiting...")
                await asyncio.sleep(10) # Wait for 10 seconds if queue is empty
    except KeyboardInterrupt:
        logger.info("Worker shutting down.")
    finally:
        # Graceful shutdown
        logger.info("Closing service connections...")
        cosmos_service.close()
        await blob_service.close()
        await queue_service.close()
        await gemini_client.close()

if __name__ == "__main__":
    asyncio.run(main())