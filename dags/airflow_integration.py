# Airflow DAG Definition
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago

import json
import time
import uuid
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multimodal_rag_system import (
    initialize_system,  
    update_metadata_from_chromadb,
    save_repo_metadata,
    load_repo_metadata,
    check_repo_for_changes,     
    process_repository,
    DEFAULT_CHROMA_PATH,
    logger
)

# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'multimodal_rag_system',
    default_args=default_args,
    description='Multimodal RAG Repository Processing Pipeline',
    schedule_interval=timedelta(hours=12),
    start_date=days_ago(1),
    catchup=False,
    tags=['multimodal', 'rag', 'nlp'],
)

# Task 1: Update metadata
def update_repo_metadata_task():
    """Task to update repository metadata from ChromaDB"""
    # Initialize the system
    initialize_system()
    
    # Update metadata
    metadata = update_metadata_from_chromadb()
    
    # Check for new repository submissions from the Streamlit app
    try:
        new_repo = Variable.get("new_repository_submission", default_var=None)
        if new_repo:
            # Parse the JSON
            repo_data = json.loads(new_repo)
            repo_url = repo_data.get("repo_url")
            process_immediately = repo_data.get("process_immediately", False)
            
            # Generate repo_id based on URL
            if "github.com/" in repo_url:
                parts = repo_url.split("github.com/")[1].split("/")
            else:
                parts = repo_url.split("/")
            
            if len(parts) >= 2:
                owner = parts[0].strip()
                repo_name = parts[1].split('.git')[0].strip()
                repo_id = f"{owner}_{repo_name}"
                repo_id = ''.join(c if c.isalnum() or c in ['_', '-', '.'] else '_' for c in repo_id)
                if not repo_id[0].isalnum():
                    repo_id = 'r' + repo_id
                if not repo_id[-1].isalnum():
                    repo_id = repo_id + 'r'
                if len(repo_id) > 60:
                    repo_id = repo_id[:60]
            else:
                repo_id = f"repo_{uuid.uuid4().hex[:8]}"
            
            # Add to metadata if not already present
            if repo_id not in metadata:
                metadata[repo_id] = {
                    "repo_url": repo_url,
                    "last_commit": None,
                    "last_processed": None,
                    "status": "submitted"
                }
                save_repo_metadata(metadata)
            
            # If immediate processing requested, return this repo_id for the next task
            if process_immediately:
                # Clear the variable
                Variable.set("new_repository_submission", "")
                return [repo_id]
            
            # Clear the variable
            Variable.set("new_repository_submission", "")
    except Exception as e:
        logger.error(f"Error processing new repository submission: {e}")
    
    # Return all repo IDs for checking
    return list(metadata.keys())

# Task 2: Check repositories for changes
def check_repo_changes_task(**context):
    """Task to check repositories for changes"""
    repo_ids = context['ti'].xcom_pull(task_ids='update_repo_metadata')
    changed_repos = []

    # If no repos found, return empty list
    if not repo_ids:
        return []

    metadata = load_repo_metadata()
    for repo_id in repo_ids:
        repo_data = metadata.get(repo_id, {})
        repo_url = repo_data.get("repo_url")
        last_commit = repo_data.get("last_commit")
        
        # Update status to "checking"
        metadata[repo_id]["status"] = "checking"
        save_repo_metadata(metadata)

        has_changed, latest_commit, changed_files = check_repo_for_changes(repo_url, last_commit)

        if has_changed:
            changed_repos.append({
                "repo_id": repo_id,
                "repo_url": repo_url,
                "latest_commit": latest_commit,
                "changed_files": changed_files
            })
            # Update status to "changes_detected"
            metadata[repo_id]["status"] = "changes_detected"
        else:
            # Update status to "up_to_date"
            metadata[repo_id]["status"] = "up_to_date"
            
    save_repo_metadata(metadata)
    return changed_repos

# Task 3: Process repositories
def process_changed_repos_task(**context):
    """Task to process repositories with changes"""
    changed_repos = context['ti'].xcom_pull(task_ids='check_repo_changes')
    results = []
    
    # If no repos to process, return empty list
    if not changed_repos:
        return []
        
    metadata = load_repo_metadata()
    
    for repo in changed_repos:
        try:
            repo_id = repo["repo_id"]
            repo_url = repo["repo_url"]
            changed_files = repo["changed_files"]
            
            # Update status to "processing"
            metadata[repo_id]["status"] = "processing"
            save_repo_metadata(metadata)
            
            # Process the repository
            result = process_repository(repo_url, DEFAULT_CHROMA_PATH, changed_files)
            
            # Update metadata
            metadata[repo_id]["last_commit"] = repo["latest_commit"]
            metadata[repo_id]["last_processed"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            metadata[repo_id]["status"] = "processed"
            metadata[repo_id]["repo_stats"] = {
                "text_files": result.get("text_files", 0),
                "image_files": result.get("image_files", 0),
                "video_files": result.get("video_files", 0),
                "chunks_processed": result.get("chunks_processed", 0)
            }
            save_repo_metadata(metadata)
            
            results.append({
                "repo_id": repo_id,
                "status": "success",
                "processed_files": result.get("text_files", 0) + result.get("image_files", 0) + result.get("video_files", 0)
            })
        except Exception as e:
            logger.error(f"Error processing repository {repo.get('repo_id', 'unknown')}: {str(e)}", exc_info=True)
            # Update status to "error"
            try:
                metadata[repo["repo_id"]]["status"] = "error"
                metadata[repo["repo_id"]]["error_message"] = str(e)
                save_repo_metadata(metadata)
            except:
                pass
                
            results.append({
                "repo_id": repo.get("repo_id", "unknown"),
                "status": "error",
                "error": str(e)
            })
    
    return results

# Task 4: Generate report
def generate_report_task(**context):
    """Task to generate a processing report"""
    processed_repos = context['ti'].xcom_pull(task_ids='process_changed_repos')
    
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_repos_processed": len(processed_repos),
        "results": processed_repos
    }
    
    # Save the report to a variable for Streamlit to access
    Variable.set("last_processing_report", json.dumps(report))
    
    return report

# Manually trigger a specific repository
def process_specific_repo_task(repo_id):
    """Task to process a specific repository on demand"""
    metadata = load_repo_metadata()
    
    if repo_id not in metadata:
        return {"status": "error", "message": f"Repository {repo_id} not found in metadata"}
    
    repo_data = metadata[repo_id]
    repo_url = repo_data["repo_url"]
    
    # Update status to "processing"
    metadata[repo_id]["status"] = "processing"
    save_repo_metadata(metadata)
    
    try:
        # Process the repository (all files)
        result = process_repository(repo_url, DEFAULT_CHROMA_PATH)
        
        # Get the latest commit
        _, latest_commit, _ = check_repo_for_changes(repo_url, None)
        
        # Update metadata
        metadata[repo_id]["last_commit"] = latest_commit
        metadata[repo_id]["last_processed"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        metadata[repo_id]["status"] = "processed"
        metadata[repo_id]["repo_stats"] = {
            "text_files": result.get("text_files", 0),
            "image_files": result.get("image_files", 0),
            "video_files": result.get("video_files", 0),
            "chunks_processed": result.get("chunks_processed", 0)
        }
        save_repo_metadata(metadata)
        
        return {
            "repo_id": repo_id,
            "status": "success",
            "processed_files": result.get("text_files", 0) + result.get("image_files", 0) + result.get("video_files", 0)
        }
    except Exception as e:
        logger.error(f"Error processing repository {repo_id}: {str(e)}", exc_info=True)
        # Update status to "error"
        metadata[repo_id]["status"] = "error"
        metadata[repo_id]["error_message"] = str(e)
        save_repo_metadata(metadata)
        
        return {
            "repo_id": repo_id,
            "status": "error",
            "error": str(e)
        }

# Define the tasks
t1 = PythonOperator(
    task_id='update_repo_metadata',
    python_callable=update_repo_metadata_task,
    dag=dag,
)

t2 = PythonOperator(
    task_id='check_repo_changes',
    python_callable=check_repo_changes_task,
    provide_context=True,
    dag=dag,
)

t3 = PythonOperator(
    task_id='process_changed_repos',
    python_callable=process_changed_repos_task,
    provide_context=True,
    dag=dag,
)

t4 = PythonOperator(
    task_id='generate_report',
    python_callable=generate_report_task,
    provide_context=True,
    dag=dag,
)

# Set dependencies
t1 >> t2 >> t3 >> t4

# Helper functions for Streamlit integration
def get_repo_status(repo_id):
    """Get the current status of a repository"""
    metadata = load_repo_metadata()
    
    if repo_id not in metadata:
        return {"status": "not_found"}
    
    return {
        "status": metadata[repo_id].get("status", "unknown"),
        "last_processed": metadata[repo_id].get("last_processed"),
        "last_commit": metadata[repo_id].get("last_commit"),
        "stats": metadata[repo_id].get("repo_stats", {}),
        "error_message": metadata[repo_id].get("error_message")
    }

def submit_repository_for_processing(repo_url, process_immediately=False):
    """Submit a repository for processing via Airflow Variable"""
    # Create the submission data
    submission = {
        "repo_url": repo_url,
        "process_immediately": process_immediately,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    # Set the Airflow Variable for the DAG to pick up
    Variable.set("new_repository_submission", json.dumps(submission))
    
    # If immediate processing requested, trigger the DAG
    if process_immediately:
        from airflow.api.client.local_client import Client
        c = Client(None, None)
        c.trigger_dag("multimodal_rag_system")
    
    return {"status": "submitted", "repository": repo_url}
