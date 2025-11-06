import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any

class DocumentVisualizer:
    def __init__(self):
        self.colors = {
            'text': (0, 255, 0),
            'table': (255, 0, 0),
            'form': (0, 0, 255),
            'header': (255, 255, 0),
            'footer': (0, 255, 255),
            'entity': (255, 0, 255)
        }
    
    def draw_layout_analysis(self, image: np.ndarray, layout_analysis: Dict[str, Any]) -> np.ndarray:
        viz_image = image.copy()
        
        for region_type, regions in layout_analysis['regions_by_type'].items():
            color = self.colors.get(region_type, (128, 128, 128))
            for region in regions:
                bbox = region['bbox']
                cv2.rectangle(viz_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                label = f"{region_type}"
                cv2.putText(viz_image, label, (bbox[0], bbox[1]-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return viz_image
    
    def draw_entities(self, image: np.ndarray, entities: List[Dict[str, Any]]) -> np.ndarray:
        viz_image = image.copy()
        
        for entity in entities:
            if 'bbox' in entity:
                bbox = entity['bbox']
                cv2.rectangle(viz_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                            self.colors['entity'], 2)
                
                label = f"{entity['type']}: {entity['text']}"
                cv2.putText(viz_image, label, (bbox[0], bbox[1]-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['entity'], 2)
        
        return viz_image
    
    def create_analysis_report(self, document_analysis: Dict[str, Any], 
                             output_path: str = None) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        if 'layout_viz' in document_analysis:
            axes[0,0].imshow(cv2.cvtColor(document_analysis['layout_viz'], cv2.COLOR_BGR2RGB))
            axes[0,0].set_title('Layout Analysis')
            axes[0,0].axis('off')
        
        if 'entities_viz' in document_analysis:
            axes[0,1].imshow(cv2.cvtColor(document_analysis['entities_viz'], cv2.COLOR_BGR2RGB))
            axes[0,1].set_title('Entity Extraction')
            axes[0,1].axis('off')
        
        doc_type = document_analysis.get('document_type', {})
        if isinstance(doc_type, dict):
            axes[1,0].bar(doc_type.get('all_probabilities', {}).keys(), 
                         doc_type.get('all_probabilities', {}).values())
            axes[1,0].set_title('Document Type Probabilities')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        entities = document_analysis.get('entities', [])
        if entities:
            entity_types = [e['type'] for e in entities]
            type_counts = {t: entity_types.count(t) for t in set(entity_types)}
            axes[1,1].bar(type_counts.keys(), type_counts.values())
            axes[1,1].set_title('Entity Type Distribution')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig