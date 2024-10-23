from langchain_core.tools import BaseTool
import pandas as pd
import json

class Respond(BaseTool):
    name: str = "respond"
    description: str = (
        "respond(reply: str, query: str, ga_result: dict) -> str - "
        "Dynamically formats and responds to the given analytics data based on the original query."
    )

    def _run(self, reply: str, query: str, ga_result: str) -> str:
        try:
            ga_results = json.loads(ga_result) if isinstance(ga_result, str) else ga_result
            
            if not ga_results:
                return f"No data available for query: '{query}'"
            
            response = []
            if reply:
                response.append(reply)
            
            for idx, result in ga_results.items():
                if "error" in result:
                    response.append(f"Error in query {idx}: {result['error']}")
                    continue
                    
                if not result.get('data'):
                    response.append(f"No data found for query {idx}")
                    continue
                
                df = pd.DataFrame(result['data'])
                
                if df.empty:
                    response.append("No data available for the specified period.")
                    continue
                
                response.append(f"Data for period: {', '.join(result['metadata']['date_ranges'])}")
                
                table = df.to_string(index=False)
                response.extend(["", table])
                
                if result.get('totals'):
                    response.append("\nTotals:")
                    for metric, value in result['totals'].items():
                        response.append(f"{metric}: {value:,.2f}")
            
            return "\n".join(response)
            
        except Exception as e:
            return f"Error processing results: {str(e)}"

respond = Respond()