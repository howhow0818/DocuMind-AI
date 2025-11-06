import json
import pandas as pd
from typing import Dict, Any, List
import os

class FileHandler:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_analysis_results(self, results: Dict[str, Any], filename: str) -> str:
        filepath = os.path.join(self.output_dir, f"{filename}_analysis.json")
        
        serializable_results = self.make_serializable(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def make_serializable(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self.make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return self.make_serializable(obj.__dict__)
        else:
            return obj
    
    def save_table_data(self, tables: Dict[str, pd.DataFrame], filename: str) -> str:
        table_dir = os.path.join(self.output_dir, "tables")
        os.makedirs(table_dir, exist_ok=True)
        
        for table_name, df in tables.items():
            csv_path = os.path.join(table_dir, f"{filename}_{table_name}.csv")
            df.to_csv(csv_path, index=False)
        
        return table_dir
    
    def save_visualizations(self, visualizations: Dict[str, Any], filename: str) -> str:
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        for viz_name, viz_data in visualizations.items():
            if isinstance(viz_data, plt.Figure):
                viz_path = os.path.join(viz_dir, f"{filename}_{viz_name}.png")
                viz_data.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close(viz_data)
        
        return viz_dir