# src/api/v1/endpoints/jobs.py (Updated)
# ... (imports)
from src.services.queue_service import StorageQueueService # <-- NEW
from src.dependencies import get_queue_service # <-- NEW

@router.post(...)
async def create_ocr_job(
    # ... (other dependencies)
    queue_service: StorageQueueService = Depends(get_queue_service), # <-- NEW
):
    # ... (file validation and resource creation logic)

    # --- THIS IS THE FINAL STEP IN THE ENDPOINT ---
    # 3. Enqueue the job for the worker
    try:
        await queue_service.send_message(str(job_id))
        logger.info(f"Successfully enqueued job {job_id}.")
    except Exception as e:
        logger.error(f"Failed to enqueue job {job_id}: {e}", exc_info=True)
        # This is a critical failure. The job exists but won't be processed.
        # You might want to add alerting or cleanup logic here.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Job created but failed to be queued for processing. Please contact support.",
        )

    return JobCreationResponse(job_id=job_id)

# ... (rest of the file)