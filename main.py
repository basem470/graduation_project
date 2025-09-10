from agents.sales_agent.salesAgent import SalesReactAgent
from agents.analytics_agent.analyticsAgent import AnalyticsReActAgent


sales_agent=SalesReactAgent()
sales_agent.query("Hello")

analytics_agent = AnalyticsReActAgent()
analytics_agent.query("what are the top selling products?")