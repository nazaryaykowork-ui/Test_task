import logging
from typing import Optional, List, Dict, Any
import pandas as pd
from google.cloud import bigquery
from google.cloud.bigquery.retry import DEFAULT_RETRY

# Fail fast: the library's default retries run for up to 10 minutes per request and 40 per
# job. The caller (agent/warehouse.py) owns the retry policy, so each request gets a short
# deadline and failed jobs are not re-run here.
REQUEST_RETRY = DEFAULT_RETRY.with_timeout(20)
REQUEST_TIMEOUT_S = 10


class BigQueryRunner:
    """A lean BigQuery client for executing SQL queries and returning DataFrame results."""
    
    def __init__(self, project_id: Optional[str] = None, dataset_id: Optional[str] = "bigquery-public-data.thelook_ecommerce") -> None:
        """Initialize BigQuery client.
        
        Args:
            project_id: Google Cloud project ID. If None, uses default credentials.
            dataset_id: BigQuery dataset ID. If None, uses default dataset.
        """
        logging.info("Initializing BigQuery client")
        try:
            self.client = bigquery.Client(project=project_id)
            self.dataset_id = dataset_id
            logging.info(f"BigQuery client initialized for dataset: {self.dataset_id}")
        except Exception as e:
            logging.error(f"Failed to initialize BigQuery client: {str(e)}")
            raise
    
    def execute_query(
        self,
        sql_query: str,
        job_config: Optional[bigquery.QueryJobConfig] = None,
        max_results: Optional[int] = None,
    ) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame.
        
        Args:
            sql_query: The SQL query to execute.
            job_config: Optional job settings (bytes-billed limit, timeout).
            max_results: Optional cap on the number of rows fetched.
            
        Returns:
            DataFrame containing the query results.
            
        Raises:
            Exception: If query execution fails.
        """
        try:
            logging.info(f"Executing BigQuery query")
            query_job = self.client.query(
                sql_query, job_config=job_config, retry=REQUEST_RETRY, timeout=REQUEST_TIMEOUT_S, job_retry=None
            )
            df = query_job.result(max_results=max_results, retry=REQUEST_RETRY, job_retry=None).to_dataframe()
            logging.info(f"Query completed successfully, returned {len(df)} rows")
            return df
        except Exception as e:
            logging.error(f"BigQuery execution failed: {str(e)}")
            raise 

    def dry_run(self, sql_query: str) -> int:
        """Validate a query without running it and return the bytes it would process."""
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        job = self.client.query(sql_query, job_config=job_config, retry=REQUEST_RETRY, timeout=REQUEST_TIMEOUT_S)
        return job.total_bytes_processed

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get schema information for a specific table.
        
        Args:
            table_name: Name of the table (orders, order_items, products, users).
            
        Returns:
            List of dictionaries containing column information.
        """
        try:
            table_ref = f"{self.dataset_id}.{table_name}"
            table = self.client.get_table(table_ref, retry=REQUEST_RETRY, timeout=REQUEST_TIMEOUT_S)
            schema_info = []
            for field in table.schema:
                schema_info.append({
                    "name": field.name,
                    "type": field.field_type,
                    "mode": field.mode,
                    "description": field.description or ""
                })
            logging.info(f"Retrieved schema for table {table_name}")
            return schema_info
        except Exception as e:
            logging.error(f"Failed to get schema for table {table_name}: {str(e)}")
            raise        