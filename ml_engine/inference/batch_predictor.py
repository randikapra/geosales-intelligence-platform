"""
Batch Prediction Processing for SFA ML Models
Handles large-scale batch predictions with parallel processing, progress tracking, and error handling
"""
import os
import sys
import logging
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import pickle
import json
from pathlib import Path
import time
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Import predictor
from predictor import UniversalPredictor, create_predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BatchJob:
    """Batch prediction job configuration"""
    job_id: str
    model_name: str
    input_data_path: str
    output_path: str
    data_type: str = 'mixed'
    chunk_size: int = 1000
    parallel_workers: int = 4
    created_at: str = None
    status: str = 'pending'  # pending, running, completed, failed
    progress: float = 0.0
    total_records: int = 0
    processed_records: int = 0
    error_message: str = None
    execution_time: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

class BatchJobManager:
    """Manages batch prediction jobs"""
    
    def __init__(self, job_store_path: str = "./batch_jobs/"):
        self.job_store_path = Path(job_store_path)
        self.job_store_path.mkdir(exist_ok=True)
        self.active_jobs = {}
        self.load_jobs()
    
    def save_job(self, job: BatchJob):
        """Save job to disk"""
        job_file = self.job_store_path / f"{job.job_id}.json"
        with open(job_file, 'w') as f:
            json.dump(asdict(job), f, indent=2)
    
    def load_jobs(self):
        """Load all jobs from disk"""
        for job_file in self.job_store_path.glob("*.json"):
            try:
                with open(job_file, 'r') as f:
                    job_data = json.load(f)
                    job = BatchJob(**job_data)
                    self.active_jobs[job.job_id] = job
            except Exception as e:
                logger.error(f"Failed to load job {job_file}: {e}")
    
    def create_job(self, model_name: str, input_data_path: str, output_path: str,
                   data_type: str = 'mixed', chunk_size: int = 1000,
                   parallel_workers: int = 4) -> BatchJob:
        """Create a new batch job"""
        job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        job = BatchJob(
            job_id=job_id,
            model_name=model_name,
            input_data_path=input_data_path,
            output_path=output_path,
            data_type=data_type,
            chunk_size=chunk_size,
            parallel_workers=parallel_workers
        )
        
        self.active_jobs[job_id] = job
        self.save_job(job)
        logger.info(f"Created batch job: {job_id}")
        return job
    
    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get job by ID"""
        return self.active_jobs.get(job_id)
    
    def update_job_status(self, job_id: str, status: str, progress: float = None,
                         processed_records: int = None, error_message: str = None):
        """Update job status"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = status
            if progress is not None:
                job.progress = progress
            if processed_records is not None:
                job.processed_records = processed_records
            if error_message is not None:
                job.error_message = error_message
            
            self.save_job(job)
    
    def list_jobs(self, status_filter: str = None) -> List[BatchJob]:
        """List all jobs, optionally filtered by status"""
        jobs = list(self.active_jobs.values())
        if status_filter:
            jobs = [job for job in jobs if job.status == status_filter]
        return sorted(jobs, key=lambda x: x.created_at, reverse=True)

class BatchPredictor:
    """High-performance batch prediction engine"""
    
    def __init__(self, predictor: UniversalPredictor = None, 
                 job_manager: BatchJobManager = None):
        self.predictor = predictor or create_predictor()
        self.job_manager = job_manager or BatchJobManager()
        self.supported_formats = ['.csv', '.xlsx', '.parquet', '.json']
    
    def load_data(self, file_path: str, chunk_size: int = None) -> Union[pd.DataFrame, pd.io.parsers.readers.TextFileReader]:
        """Load data from various formats"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        if file_path.suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        try:
            if file_path.suffix == '.csv':
                if chunk_size:
                    return pd.read_csv(file_path, chunksize=chunk_size)
                else:
                    return pd.read_csv(file_path)
            elif file_path.suffix == '.xlsx':
                return pd.read_excel(file_path)
            elif file_path.suffix == '.parquet':
                return pd.read_parquet(file_path)
            elif file_path.suffix == '.json':
                return pd.read_json(file_path)
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            raise
    
    def save_results(self, results: pd.DataFrame, output_path: str, format_type: str = 'csv'):
        """Save prediction results"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format_type == 'csv':
                results.to_csv(output_path, index=False)
            elif format_type == 'xlsx':
                results.to_excel(output_path, index=False)
            elif format_type == 'parquet':
                results.to_parquet(output_path, index=False)
            elif format_type == 'json':
                results.to_json(output_path, orient='records', indent=2)
            
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results to {output_path}: {e}")
            raise
    
    def process_chunk(self, chunk_data: pd.DataFrame, model_name: str, 
                     data_type: str, chunk_id: int) -> Dict[str, Any]:
        """Process a single chunk of data"""
        try:
            start_time = time.time()
            
            # Make predictions
            predictions = self.predictor.predict(model_name, chunk_data, data_type)
            
            # Prepare results
            results_df = chunk_data.copy()
            
            # Add predictions to results
            if isinstance(predictions['predictions'], list):
                if len(predictions['predictions']) == len(chunk_data):
                    results_df['prediction'] = predictions['predictions']
                else:
                    # Handle multi-class predictions
                    for i, pred in enumerate(predictions['predictions']):
                        results_df[f'prediction_{i}'] = pred
            
            # Add probabilities if available
            if 'probabilities' in predictions:
                probs = predictions['probabilities']
                if len(probs) == len(chunk_data):
                    if len(probs[0]) == 2:  # Binary classification
                        results_df['probability'] = [p[1] for p in probs]
                    else:  # Multi-class
                        for i in range(len(probs[0])):
                            results_df[f'probability_class_{i}'] = [p[i] for p in probs]
            
            # Add confidence scores if available
            if 'confidence_scores' in predictions:
                results_df['confidence'] = predictions['confidence_scores']
            
            # Add metadata
            results_df['prediction_timestamp'] = datetime.now().isoformat()
            results_df['model_name'] = model_name
            results_df['chunk_id'] = chunk_id
            
            processing_time = time.time() - start_time
            
            return {
                'chunk_id': chunk_id,
                'results': results_df,
                'processing_time': processing_time,
                'records_processed': len(chunk_data),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Failed to process chunk {chunk_id}: {e}")
            return {
                'chunk_id': chunk_id,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'records_processed': 0,
                'status': 'failed'
            }
    
    def run_batch_job(self, job_id: str, progress_callback: Callable = None) -> Dict[str, Any]:
        """Run a batch prediction job"""
        job = self.job_manager.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        start_time = time.time()
        
        try:
            # Update job status
            self.job_manager.update_job_status(job_id, 'running', 0.0)
            
            # Load data
            logger.info(f"Loading data from {job.input_data_path}")
            data_chunks = self.load_data(job.input_data_path, job.chunk_size)
            
            # Handle single DataFrame vs chunked reader
            if isinstance(data_chunks, pd.DataFrame):
                total_records = len(data_chunks)
                chunks = [data_chunks[i:i+job.chunk_size] for i in range(0, len(data_chunks), job.chunk_size)]
            else:
                # For chunked reader, we need to process differently
                chunks = list(data_chunks)
                total_records = sum(len(chunk) for chunk in chunks)
            
            job.total_records = total_records
            self.job_manager.save_job(job)
            
            logger.info(f"Processing {total_records} records in {len(chunks)} chunks")
            
            # Process chunks in parallel
            all_results = []
            processed_records = 0
            
            if job.parallel_workers > 1:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=job.parallel_workers) as executor:
                    # Submit all tasks
                    future_to_chunk = {
                        executor.submit(self.process_chunk, chunk, job.model_name, 
                                      job.data_type, i): i 
                        for i, chunk in enumerate(chunks)
                    }
                    
                    # Process completed tasks
                    for future in tqdm(as_completed(future_to_chunk), 
                                     total=len(chunks), desc="Processing chunks"):
                        chunk_result = future.result()
                        
                        if chunk_result['status'] == 'success':
                            all_results.append(chunk_result['results'])
                            processed_records += chunk_result['records_processed']
                        else:
                            logger.error(f"Chunk {chunk_result['chunk_id']} failed: {chunk_result.get('error')}")
                        
                        # Update progress
                        progress = processed_records / total_records
                        self.job_manager.update_job_status(job_id, 'running', progress, processed_records)
                        
                        # Call progress callback if provided
                        if progress_callback:
                            progress_callback(job_id, progress, processed_records, total_records)
            
            else:
                # Sequential processing
                for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
                    chunk_result = self.process_chunk(chunk, job.model_name, job.data_type, i)
                    
                    if chunk_result['status'] == 'success':
                        all_results.append(chunk_result['results'])
                        processed_records += chunk_result['records_processed']
                    else:
                        logger.error(f"Chunk {i} failed: {chunk_result.get('error')}")
                    
                    # Update progress
                    progress = processed_records / total_records
                    self.job_manager.update_job_status(job_id, 'running', progress, processed_records)
                    
                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(job_id, progress, processed_records, total_records)
            
            # Combine all results
            if all_results:
                final_results = pd.concat(all_results, ignore_index=True)
                
                # Save results
                output_format = Path(job.output_path).suffix[1:]  # Remove the dot
                self.save_results(final_results, job.output_path, output_format)
                
                # Update job completion
                execution_time = time.time() - start_time
                job.execution_time = execution_time
                self.job_manager.update_job_status(job_id, 'completed', 1.0, processed_records)
                
                logger.info(f"Batch job {job_id} completed successfully in {execution_time:.2f} seconds")
                
                return {
                    'job_id': job_id,
                    'status': 'completed',
                    'total_records': total_records,
                    'processed_records': processed_records,
                    'execution_time': execution_time,
                    'output_path': job.output_path,
                    'results_count': len(final_results)
                }
            
            else:
                raise Exception("No successful chunks processed")
                
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)
            logger.error(f"Batch job {job_id} failed: {error_message}")
            
            job.execution_time = execution_time
            self.job_manager.update_job_status(job_id, 'failed', error_message=error_message)
            
            return {
                'job_id': job_id,
                'status': 'failed',
                'error': error_message,
                'execution_time': execution_time
            }
    
    async def run_batch_job_async(self, job_id: str, progress_callback: Callable = None) -> Dict[str, Any]:
        """Run batch job asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run_batch_job, job_id, progress_callback)
    
    def schedule_batch_job(self, model_name: str, input_data_path: str, 
                          output_path: str, **kwargs) -> str:
        """Schedule a new batch job"""
        job = self.job_manager.create_job(
            model_name=model_name,
            input_data_path=input_data_path,
            output_path=output_path,
            **kwargs
        )
        return job.job_id
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status and progress"""
        job = self.job_manager.get_job(job_id)
        if not job:
            return {'error': 'Job not found'}
        
        return {
            'job_id': job.job_id,
            'status': job.status,
            'progress': job.progress,
            'total_records': job.total_records,
            'processed_records': job.processed_records,
            'execution_time': job.execution_time,
            'created_at': job.created_at,
            'error_message': job.error_message
        }
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        job = self.job_manager.get_job(job_id)
        if job and job.status == 'running':
            self.job_manager.update_job_status(job_id, 'cancelled')
            logger.info(f"Job {job_id} cancelled")
            return True
        return False
    
    def cleanup_old_jobs(self, days_old: int = 30):
        """Clean up old completed/failed jobs"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        jobs_to_remove = []
        for job_id, job in self.job_manager.active_jobs.items():
            job_date = datetime.fromisoformat(job.created_at)
            if job_date < cutoff_date and job.status in ['completed', 'failed', 'cancelled']:
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            job_file = self.job_manager.job_store_path / f"{job_id}.json"
            if job_file.exists():
                job_file.unlink()
            del self.job_manager.active_jobs[job_id]
            logger.info(f"Cleaned up old job: {job_id}")
        
        return len(jobs_to_remove)

# Utility functions for specific SFA use cases
class SFABatchPredictor(BatchPredictor):
    """SFA-specific batch predictor with domain knowledge"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sfa_models = {
            'sales_forecast': 'sales_forecaster',
            'demand_forecast': 'demand_forecaster',
            'customer_segmentation': 'customer_segmentation',
            'churn_prediction': 'churn_prediction',
            'dealer_performance': 'dealer_performance',
            'route_optimization': 'route_optimizer',
            'territory_optimization': 'territory_optimizer'
        }
    
    def process_sales_data(self, input_path: str, output_path: str, 
                          forecast_horizon: int = 30) -> str:
        """Process sales data for forecasting"""
        job_id = self.schedule_batch_job(
            model_name='sales_forecast',
            input_data_path=input_path,
            output_path=output_path,
            data_type='sales',
            chunk_size=500
        )
        return job_id
    
    def process_customer_segmentation(self, input_path: str, output_path: str) -> str:
        """Process customer data for segmentation"""
        job_id = self.schedule_batch_job(
            model_name='customer_segmentation',
            input_data_path=input_path,
            output_path=output_path,
            data_type='customer',
            chunk_size=1000
        )
        return job_id
    
    def process_churn_prediction(self, input_path: str, output_path: str) -> str:
        """Process customer data for churn prediction"""
        job_id = self.schedule_batch_job(
            model_name='churn_prediction',
            input_data_path=input_path,
            output_path=output_path,
            data_type='customer',
            chunk_size=1000
        )
        return job_id
    
    def process_gps_route_optimization(self, input_path: str, output_path: str) -> str:
        """Process GPS data for route optimization"""
        job_id = self.schedule_batch_job(
            model_name='route_optimization',
            input_data_path=input_path,
            output_path=output_path,
            data_type='gps',
            chunk_size=200,
            parallel_workers=2  # GPS processing is more CPU intensive
        )
        return job_id

# CLI interface
def main():
    """Command line interface for batch processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SFA Batch Prediction Engine')
    parser.add_argument('--model', required=True, help='Model name to use')
    parser.add_argument('--input', required=True, help='Input data file path')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--data-type', default='mixed', help='Data type (sales, customer, gps, mixed)')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size for processing')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--job-id', help='Job ID to check status')
    parser.add_argument('--list-jobs', action='store_true', help='List all jobs')
    parser.add_argument('--cleanup', type=int, help='Clean up jobs older than N days')
    
    args = parser.parse_args()
    
    predictor = SFABatchPredictor()
    
    if args.list_jobs:
        jobs = predictor.job_manager.list_jobs()
        for job in jobs:
            print(f"Job ID: {job.job_id}, Status: {job.status}, Progress: {job.progress:.2%}")
        return
    
    if args.cleanup:
        cleaned = predictor.cleanup_old_jobs(args.cleanup)
        print(f"Cleaned up {cleaned} old jobs")
        return
    
    if args.job_id:
        status = predictor.get_job_status(args.job_id)
        print(f"Job Status: {status}")
        return
    
    # Schedule new job
    job_id = predictor.schedule_batch_job(
        model_name=args.model,
        input_data_path=args.input,
        output_path=args.output,
        data_type=args.data_type,
        chunk_size=args.chunk_size,
        parallel_workers=args.workers
    )
    
    print(f"Scheduled batch job: {job_id}")
    
    # Run the job
    def progress_callback(job_id, progress, processed, total):
        print(f"Job {job_id}: {progress:.1%} complete ({processed}/{total} records)")
    
    result = predictor.run_batch_job(job_id, progress_callback)
    print(f"Job completed: {result}")

if __name__ == "__main__":
    main()