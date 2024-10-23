from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import RunReportRequest, DateRange, Metric, Dimension, OrderBy, FilterExpression, Filter
from google.api_core.exceptions import InvalidArgument
from langchain_core.tools import BaseTool
from pydantic import Field, BaseModel
from typing import Dict, List, Any, Optional, Tuple, Union
import os
import pandas as pd
from pydantic import Field
from datetime import datetime, timedelta
import traceback
import json

class GoogleAnalyticsTool(BaseTool):
    name: str = "ga-tool"
    description: str = (
        "ga-tool(date_ranges: list, metrics: list, dimensions: list = None, "
        "property_id: str = None, order_bys: list = None, dimensional_filter: dict = None) -> str - "
        "Queries Google Analytics for specified metrics, dimensions, date ranges, order_bys, and dimensional_filter."
    )
    
    client: Optional[BetaAnalyticsDataClient] = Field(default=None)
    property_id: str = Field(default_factory=lambda: os.getenv('GA_PROPERTY_ID'))
    credentials: Optional[Credentials] = Field(default=None)
    available_dimensions: List[str] = Field(default_factory=list)
    available_metrics: List[str] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
        self.credentials = self._get_credentials()
        self._load_available_metrics_and_dimensions()
        
    def _get_credentials(self):
        creds = None
        token_file = 'token.json'
        if os.path.exists(token_file):
            creds = Credentials.from_authorized_user_file(token_file)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = Flow.from_client_secrets_file(
                    'oauth_client.json', 
                    scopes=['https://www.googleapis.com/auth/analytics.readonly']
                )
                flow.run_local_server(port=8501)
                creds = flow.credentials

            with open(token_file, 'w') as token:
                token.write(creds.to_json())

        return creds

    def _create_client(self):
        return BetaAnalyticsDataClient(credentials=self.credentials)

    def _load_available_metrics_and_dimensions(self):
        dimensions = pd.read_csv('documentation/dimensions.csv')
        metrics = pd.read_csv('documentation/metrics.csv')
        self.available_dimensions = dimensions.iloc[:, 0].to_list()
        self.available_metrics = metrics.iloc[:, 0].to_list()

    def _validate_query(self, metrics: List[str], dimensions: List[str]) -> Tuple[bool, List[str]]:
        missing = []
        for metric in metrics:
            if metric not in self.available_metrics:
                missing.append(metric)
        for dimension in dimensions:
            if dimension not in self.available_dimensions:
                missing.append(dimension)
        return len(missing) == 0, missing

    def _parse_date_ranges(self, date_ranges: Union[List[Dict[str, str]], List[str], str, Dict[str, str]]) -> List[DateRange]:
        parsed_ranges = []
        
        if isinstance(date_ranges, str):
            date_ranges = [date_ranges]
        elif isinstance(date_ranges, dict):
            date_ranges = [date_ranges]
        elif not isinstance(date_ranges, list):
            raise ValueError(f"Invalid date_ranges format: {date_ranges}")
        
        for date_range in date_ranges:
            if isinstance(date_range, str):
                if date_range == 'lastMonth':
                    today = datetime.now().date()
                    first_day_of_month = today.replace(day=1)
                    last_month_end = first_day_of_month - timedelta(days=1)
                    last_month_start = last_month_end.replace(day=1)
                    parsed_ranges.append(DateRange(
                        start_date=last_month_start.strftime('%Y-%m-%d'),
                        end_date=last_month_end.strftime('%Y-%m-%d')
                    ))
                elif date_range == 'last30days':
                    today = datetime.now().date()
                    start_date = today - timedelta(days=30)
                    parsed_ranges.append(DateRange(
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=today.strftime('%Y-%m-%d')
                    ))
                else:
                    date = self._parse_date(date_range)
                    if date:
                        parsed_ranges.append(DateRange(
                            start_date=date.strftime('%Y-%m-%d'),
                            end_date=date.strftime('%Y-%m-%d')
                        ))
            elif isinstance(date_range, dict):
                start_key = next((k for k in ['start_date', 'startDate'] if k in date_range), None)
                end_key = next((k for k in ['end_date', 'endDate'] if k in date_range), None)
                
                if start_key and end_key:
                    start_date = self._parse_date(date_range[start_key])
                    end_date = self._parse_date(date_range[end_key])
                    if start_date and end_date:
                        parsed_ranges.append(DateRange(
                            start_date=start_date.strftime('%Y-%m-%d'),
                            end_date=end_date.strftime('%Y-%m-%d')
                        ))
        
        if not parsed_ranges:
            today = datetime.now().date()
            first_day_of_month = today.replace(day=1)
            last_month_end = first_day_of_month - timedelta(days=1)
            last_month_start = last_month_end.replace(day=1)
            parsed_ranges.append(DateRange(
                start_date=last_month_start.strftime('%Y-%m-%d'),
                end_date=last_month_end.strftime('%Y-%m-%d')
            ))
        
        return parsed_ranges

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        if not date_str:
            return None
        try:
            if date_str == 'today':
                return datetime.now().date()
            elif date_str == 'yesterday':
                return (datetime.now() - timedelta(days=1)).date()
            elif date_str.endswith('daysAgo'):
                days = int(date_str[:-7])
                return (datetime.now() - timedelta(days=days)).date()
            else:
                return datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            print(f"Warning: Invalid date format: {date_str}. Using None.")
            return None
    
    def _run(self, 
            date_ranges: Union[List[Dict[str, str]], List[str], str, Dict[str, str]], 
            metrics: List[str], 
            dimensions: List[str] = None,
            property_id: Optional[str] = None,
            order_bys: Optional[List[Dict[str, str]]] = None,
            dimensional_filter: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
        
        try:
            if self.client is None:
                self.client = self._create_client()

            if property_id is None:
                property_id = self.property_id

            is_valid, missing = self._validate_query(metrics, dimensions or [])
            if not is_valid:
                return {"error": f"Invalid metrics or dimensions: {', '.join(missing)}"}

            parsed_date_ranges = self._parse_date_ranges(date_ranges)
            
            request = RunReportRequest(
                property=f"properties/{property_id}",
                date_ranges=parsed_date_ranges,
                metrics=[Metric(name=metric) for metric in metrics],
                dimensions=[Dimension(name=dim) for dim in dimensions] if dimensions else []
            )

            if order_bys:
                request.order_bys = [
                    OrderBy(
                        metric=OrderBy.MetricOrderBy(metric_name=order_by['metric']),
                        desc=order_by.get('desc', False)
                    )
                    for order_by in order_bys
                ]

            if dimensional_filter:
                request.dimension_filter = self._create_filter_expression(dimensional_filter)

            response = self.client.run_report(request)
            df = self._process_response(response)
            
            result = {
                "data": df.to_dict(orient='records'),
                "row_count": len(df),
                "metadata": {
                    "metrics": metrics,
                    "dimensions": dimensions or [],
                    "date_ranges": [f"{dr.start_date} to {dr.end_date}" for dr in parsed_date_ranges],
                    "order_bys": order_bys,
                    "dimensional_filter": dimensional_filter
                },
                "totals": df[metrics].sum().to_dict() if not df.empty else {}
            }
            
            if df.empty:
                result["warning"] = "The query returned no data."
            
            return result

        except ValueError as e:
            return {"error": f"Value Error: {str(e)}"}
        except InvalidArgument as e:
            return {"error": f"Google Analytics API Error: {str(e)}"}
        except Exception as e:
            return {
                "error": "Unexpected Error",
                "details": str(e),
                "traceback": traceback.format_exc(),
                "debug_info": {
                    "property_id": property_id,
                    "date_ranges": str(date_ranges),
                    "metrics": metrics,
                    "dimensions": dimensions,
                    "order_bys": order_bys,
                    "dimensional_filter": dimensional_filter
                }
            }

    def _create_filter_expression(self, filter_dict: Dict[str, Any]) -> FilterExpression:
        if 'andGroup' in filter_dict:
            return FilterExpression(
                and_group=FilterExpression.AndGroup(
                    expressions=[self._create_filter_expression(expr) for expr in filter_dict['andGroup']['expressions']]
                )
            )
        elif 'orGroup' in filter_dict:
            return FilterExpression(
                or_group=FilterExpression.OrGroup(
                    expressions=[self._create_filter_expression(expr) for expr in filter_dict['orGroup']['expressions']]
                )
            )
        elif 'notExpression' in filter_dict:
            return FilterExpression(
                not_expression=self._create_filter_expression(filter_dict['notExpression'])
            )
        elif 'filter' in filter_dict:
            return FilterExpression(
                filter=self._create_filter(filter_dict['filter'])
            )
        else:
            raise ValueError(f"Invalid filter structure: {filter_dict}")

    def _create_filter(self, filter_dict: Dict[str, Any]) -> Filter:
        field_name = filter_dict['fieldName']
        if 'stringFilter' in filter_dict:
            return Filter(
                field_name=field_name,
                string_filter=Filter.StringFilter(
                    match_type=filter_dict['stringFilter']['matchType'],
                    value=filter_dict['stringFilter']['value']
                )
            )
        elif 'numericFilter' in filter_dict:
            return Filter(
                field_name=field_name,
                numeric_filter=Filter.NumericFilter(
                    operation=filter_dict['numericFilter']['operation'],
                    value=Filter.NumericValue(
                        int64_value=filter_dict['numericFilter']['value']
                    )
                )
            )
        elif 'inListFilter' in filter_dict:
            return Filter(
                field_name=field_name,
                in_list_filter=Filter.InListFilter(
                    values=filter_dict['inListFilter']['values']
                )
            )
        elif 'betweenFilter' in filter_dict:
            return Filter(
                field_name=field_name,
                between_filter=Filter.BetweenFilter(
                    from_value=Filter.NumericValue(
                        int64_value=filter_dict['betweenFilter']['fromValue']
                    ),
                    to_value=Filter.NumericValue(
                        int64_value=filter_dict['betweenFilter']['toValue']
                    )
                )
            )
        else:
            raise ValueError(f"Invalid filter type in: {filter_dict}")
            
    def _process_response(self, response):
        rows = []
        for row in response.rows:
            dimension_values = [value.value for value in row.dimension_values]
            metric_values = [self._convert_metric_value(value.value) for value in row.metric_values]
            rows.append(dimension_values + metric_values)

        headers = (
            [header.name for header in response.dimension_headers] +
            [header.name for header in response.metric_headers]
        )
        
        df = pd.DataFrame(rows, columns=headers)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        return df

    def _convert_metric_value(self, value: str) -> Union[int, float]:
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
            
    def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("This tool does not support async")

ga_tool = GoogleAnalyticsTool()