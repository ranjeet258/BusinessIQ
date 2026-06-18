import duckdb
import pandas as pd
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class DuckDBManager:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = duckdb.connect(db_path)
        self.tables = {}
        
    def register_dataframe(self, table_name: str, df: pd.DataFrame):
        """Registers a pandas DataFrame as a virtual table in DuckDB."""
        try:
            self.conn.register(table_name, df)
            self.tables[table_name] = df.columns.tolist()
            logger.info(f"Registered table {table_name}")
        except Exception as e:
            logger.error(f"Error registering table {table_name}: {e}")
            raise
            
    def execute_query(self, query: str) -> pd.DataFrame:
        """Executes a SQL query against the DuckDB instance."""
        try:
            result = self.conn.execute(query).fetchdf()
            return result
        except Exception as e:
            logger.error(f"Error executing query {query}: {e}")
            raise
            
    def get_schema(self) -> str:
        """Returns the schema of all registered tables."""
        if not self.tables:
            return "No tables registered."
        schema_str = "Database Schema:\\n"
        for table, cols in self.tables.items():
            schema_str += f"- Table `{table}` with columns: {', '.join(cols)}\\n"
            try:
                sample_df = self.execute_query(f"SELECT * FROM {table} LIMIT 1")
                sample_dict = sample_df.to_dict(orient='records')
                if sample_dict:
                    schema_str += f"  Sample row: {sample_dict[0]}\\n"
            except Exception as e:
                logger.error(f"Error fetching sample for {table}: {e}")
        return schema_str
