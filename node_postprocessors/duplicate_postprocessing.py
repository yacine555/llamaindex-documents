from abc import abstractmethod 
from typing import List, Optional

from llama_index import QueryBundle
from llama_index.schema import NodeWithScore

class DuplicateRemoverNodePostprocessor:
    @abstractmethod
    def postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]) -> List[NodeWithScore]:
        
        print("Enter custom postproessor")
        unique_hashes = set()
        unique_nodes = []

        for node in nodes:
            node_nash = node.node.hash
            
            if node_nash not in unique_hashes:
                unique_hashes.add(node_nash)
                unique_nodes.append(node)

        return unique_nodes